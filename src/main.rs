use crate::bert::BertEncoder;
use crate::cf_model::CollaborativeFilteringModel;
use crate::datasets::{DataLoader, IdEncoder, TensorDataset, split_data};
use crate::server::{AppError, AppState, RecommendQuery, RecommendationResult};
use crate::types::Movie;
use anyhow::Context;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use axum::{Router, routing::get};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::loss::mse;
use candle_nn::{Optimizer, SGD, VarBuilder, VarMap};
use clap::{Parser, Subcommand};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::ScoredPoint;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;
use types::Interaction;

mod bert;
mod cf_model;
mod datasets;
mod loss;
mod metrics;
mod recommenders;
mod types;

mod server;
mod vector_store;

fn read_interactions(path: &str) -> Result<Vec<Interaction>, Box<dyn std::error::Error>> {
    let mut data =
        csv::Reader::from_path(PathBuf::from(path)).expect("ファイルパスが開けませんでした");
    let data = data
        .deserialize()
        .map(|x| x.expect("行をパースできませんでした"))
        .collect::<Vec<Interaction>>();
    Ok(data)
}

fn read_movie(path: &str) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    let mut data =
        csv::Reader::from_path(PathBuf::from(path)).expect("ファイルパスが開けませんでした");
    let mut mapper = HashMap::new();
    for result in data.deserialize() {
        let record: Movie = result?;
        mapper.insert(record.item_id, record.title);
    }
    Ok(mapper)
}

fn precompute_embeddings(
    item_encoder: &IdEncoder,
    id2title: &HashMap<String, String>,
    bert: &BertEncoder,
    batch_size: usize,
) -> anyhow::Result<Tensor> {
    let n_items = item_encoder.len();
    let mut all_embeddings = Vec::new();

    let mut titles = Vec::new();
    for i in 0..n_items {
        let item_id = item_encoder.decode(i).unwrap_or("Unknown");
        let title = id2title
            .get(item_id)
            .cloned()
            .unwrap_or_else(|| "Unknown".to_string());
        titles.push(title.clone());
    }

    // 2. バッチごとに処理する
    for batch_titles in titles.chunks(batch_size) {
        let batch_vec = batch_titles.to_vec();
        let embedding = bert.encode(batch_vec)?;
        all_embeddings.push(embedding);
    }
    Tensor::cat(&all_embeddings, 0).map_err(|e| anyhow::anyhow!(e))
}

async fn load_movies() -> anyhow::Result<()> {
    let device = Device::new_metal(0)?;

    let path = "data/movielens_small/ratings.csv";
    println!("Loading data from {path}");

    let all_interactions = read_interactions(path).expect("データが読めませんでした");
    println!("Total interactions: {}", all_interactions.len());

    let (train_data, test_data) = split_data(all_interactions, 0.8);
    println!("Train: {}, Test: {}", train_data.len(), test_data.len());

    // Encoderの作成 (学習データに含まれるIDのみを知っている状態にする)
    let user_encoder = IdEncoder::new(train_data.iter().map(|x| &x.user_id));
    let item_encoder = IdEncoder::new(train_data.iter().map(|x| &x.item_id));

    println!(
        "Users: {}, Items: {}",
        user_encoder.len(),
        item_encoder.len()
    );

    // Datasetの作成
    println!("--- Content-based Recommendation ---");

    println!("Loading BERT model...");
    let bert = BertEncoder::new(&device)?;

    println!("Generating embeddings for {} movies...", item_encoder.len());
    let id2title =
        read_movie("data/movielens_small/movies.csv").expect("Movieファイルを読み込めませんでした");
    let item_embeddings = precompute_embeddings(&item_encoder, &id2title, &bert, 32)?;
    println!("Embedding shape: {:?}", item_embeddings.shape());

    println!("--- Vector Search with Qdrant ---");

    let qdrant_url = "http://localhost:6334";
    let qdrant_client = vector_store::init_qdrant(qdrant_url).await?;

    vector_store::upsert_movies(&qdrant_client, &item_encoder, &id2title, &item_embeddings).await?;
    Ok(())
}

async fn search_movie(limit: u64, query_text: &str) -> anyhow::Result<Vec<ScoredPoint>> {
    // load_qdrant().await?;
    let device = Device::new_metal(0)?;

    let path = "data/movielens_small/ratings.csv";
    println!("Loading data from {path}");

    let all_interactions = read_interactions(path).expect("データが読めませんでした");
    println!("Total interactions: {}", all_interactions.len());

    // Encoderの作成 (学習データに含まれるIDのみを知っている状態にする)
    let user_encoder = IdEncoder::new(all_interactions.iter().map(|x| &x.user_id));
    let item_encoder = IdEncoder::new(all_interactions.iter().map(|x| &x.item_id));

    println!(
        "Users: {}, Items: {}",
        user_encoder.len(),
        item_encoder.len()
    );

    println!("Loading BERT model...");
    let bert = BertEncoder::new(&device)?;

    println!("\nQuery: \"{}\"", query_text);

    let query_embedding = bert.encode(vec![query_text.to_string()])?;

    let query_vector = query_embedding.to_vec2::<f32>()?[0].clone();

    let qdrant_url = "http://localhost:6334";
    let qdrant_client = Qdrant::from_url(qdrant_url)
        .build()
        .expect("Failed to create Qdrant client");
    let result = vector_store::search_movies(&qdrant_client, query_vector, limit).await?;
    Ok(result)
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "data/movielens_small/ratings.csv")]
    ratings_path: String,

    #[arg(long, default_value = "data/movielens_small/movies.csv")]
    movies_path: String,

    #[arg(long, default_value = "http://localhost:6334")]
    qdrant_url: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// MFモデルの学習のパラメータ
    Train {
        #[arg(long, default_value_t = 0.5)]
        learning_rate: f64,

        #[arg(long, default_value_t = 50)]
        epochs: usize,

        #[arg(long, default_value_t = 32)]
        embedding_dims: usize,

        #[arg(long, default_value_t = 64)]
        batch_size: usize,

        #[arg(long, default_value_t = 0.0001)]
        lambda_: f64,
    },
    /// キーワードで映画を検索
    Search {
        #[arg(short, long)]
        query: String,
    },
    Server,
}

#[derive(Debug, Clone)]
struct TrainingConfig {
    pub learning_rate: f64,
    pub epochs: usize,
    pub embedding_dims: usize,
    pub batch_size: usize,
    pub lambda_: f64,
}

fn prepare_data(
    rating_path: &str,
    device: &Device,
) -> anyhow::Result<(TensorDataset, TensorDataset, IdEncoder, IdEncoder)> {
    println!("Loading data from {rating_path}");

    let all_interactions = read_interactions(rating_path).expect("データが読めませんでした");
    println!("Total interactions: {}", all_interactions.len());

    let (train_data, test_data) = split_data(all_interactions, 0.8);
    println!("Train: {}, Test: {}", train_data.len(), test_data.len());

    // Encoderの作成 (学習データに含まれるIDのみを知っている状態にする)
    let user_encoder = IdEncoder::new(train_data.iter().map(|x| &x.user_id));
    let item_encoder = IdEncoder::new(train_data.iter().map(|x| &x.item_id));

    let train_dataset = TensorDataset::new(&train_data, &user_encoder, &item_encoder, device)?;
    let test_dataset = TensorDataset::new(&test_data, &user_encoder, &item_encoder, device)?;
    Ok((train_dataset, test_dataset, user_encoder, item_encoder))
}

fn train_model(
    config: &TrainingConfig,
    train_dataset: &TensorDataset,
    test_dataset: &TensorDataset,
    user_encoder: &IdEncoder,
    item_encoder: &IdEncoder,
    device: &Device,
) -> anyhow::Result<CollaborativeFilteringModel> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);

    let model = CollaborativeFilteringModel::new(
        vb,
        user_encoder.len(),
        item_encoder.len(),
        config.embedding_dims,
    )?;

    let mut opt = SGD::new(varmap.all_vars(), config.learning_rate)?;

    println!("Start training...");

    for epoch in 1..=config.epochs {
        let mut total_train_loss = 0.0;
        let mut train_steps = 0;

        let train_loader = DataLoader::new(train_dataset.clone(), config.batch_size);
        for (u, i, r) in train_loader {
            let logits = model.forward(&u, &i)?;
            let mut loss = mse(&logits, &r)?;
            let mut l2_reg = Tensor::zeros((), DType::F32, device)?;

            for var in varmap.all_vars() {
                l2_reg = (l2_reg + var.sqr()?.sum_all()?)?;
            }
            loss = (loss + (l2_reg * config.lambda_)?)?;

            opt.backward_step(&loss)?;
            total_train_loss += loss.to_scalar::<f32>()?;
            train_steps += 1;
        }

        let mut total_test_loss = 0.0;
        let mut test_steps = 0;

        let test_loader = DataLoader::new(test_dataset.clone(), config.batch_size);
        for (u, i, r) in test_loader {
            let logits = model.forward(&u, &i)?;
            let loss = mse(&logits, &r)?;
            total_test_loss += loss.to_scalar::<f32>()?;
            test_steps += 1;
        }

        let avg_train_loss = total_train_loss / train_steps as f32;
        let avg_test_loss = total_test_loss / test_steps as f32;

        println!(
            "Epoch {:>2} | Train Loss: {:.4} | Test Loss: {:.4}",
            epoch, avg_train_loss, avg_test_loss,
        );
    }
    varmap
        .save("model.safetensors")
        .expect("重みの保存に失敗しました");
    user_encoder.save("user_encoder.json")?;
    item_encoder.save("item_encoder.json")?;
    Ok(model)
}


async fn hybrid_recommendation(
    State(state): State<Arc<AppState>>,
    Query(query): Query<RecommendQuery>,
) -> std::result::Result<Json<Vec<RecommendationResult>>, AppError> {
    let item_encoder = &state.item_encoder;
    let user_encoder = &state.user_encoder;

    let id2title = &state.id2title;
    let qdrant_client = &state.qdrant_client;
    let embedding_model = &state.embedding_model;
    let ranking_model = &state.ranking_model;

    let device = ranking_model.device().context("Cannot get device")?;
    let query_text = query.query.clone();
    let query_embedding = embedding_model.encode(vec![query_text]).context("Embedding generated failed")?;

    let query_vector = query_embedding.to_vec2::<f32>().context("Cannot convert tensor to vec")?[0].clone();

    let candidates =
        vector_store::search_movies(qdrant_client, query_vector, query.limit as u64).await?;

    // 候補の内部IDを抽出 (U32)
    let candidate_ids: Vec<u32> = candidates
        .iter()
        .map(|point| {
            match point
                .id
                .as_ref()
                .context("Cannot get Id")?
                .point_id_options
                .as_ref()
                .context("Cannot get id options")?
            {
                qdrant_client::qdrant::point_id::PointIdOptions::Num(num) => Ok(*num as u32),
                _ => Ok(0),
            }
        })
        .collect::<anyhow::Result<Vec<u32>>>()?;

    let user_idx = user_encoder
        .encode(&query.user_id)
        .ok_or_else(|| AppError::UserNotFound(query.user_id.clone()))?;
    let n_items = candidates.len();

    let user_indices: Vec<u32> = vec![user_idx as u32; n_items];
    let item_indices: Vec<u32> = candidate_ids.clone();

    let user_input = Tensor::from_vec(user_indices, n_items, device).context("Cannot Generate User Tensor")?;
    let item_input = Tensor::from_vec(item_indices, n_items, device).context("Cannot Generate Item Tensor")?;

    let scores = ranking_model
        .forward(&user_input, &item_input)
        .context("Model Inference failed")?
        .to_vec1::<f32>().context("Cannot convert ranking score to vec")?;
    let mut scored_items: Vec<(usize, f32)> =
        scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
    scored_items.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut results = Vec::new();
    for (idx, score) in scored_items.iter().take(query.limit) {
        let real_ids = candidate_ids[*idx];
        let original_id = item_encoder
            .decode(real_ids as usize)
            .context("Cannot decode item _id")?;

        let title = id2title
            .get(original_id)
            .map(|s| s.as_str())
            .unwrap_or("Unknown Title");
        results.push(RecommendationResult {
            score: *score as f64,
            title: title.to_string(),
            item_id: original_id.to_string(),
        })
    }
    Ok(Json(results))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let device = Device::new_metal(0)?; // Device::new_metal(0).unwrap_or(Device::Cpu);
    match args.command {
        Commands::Train {
            learning_rate,
            epochs,
            embedding_dims,
            batch_size,
            lambda_,
        } => {
            println!("--- Training Mode ---");
            let config = TrainingConfig {
                learning_rate,
                epochs,
                embedding_dims,
                batch_size,
                lambda_,
            };

            let (train_dataset, test_dataset, user_encoder, item_encoder) =
                prepare_data(&args.ratings_path, &device)?;
            let _model = train_model(
                &config,
                &train_dataset,
                &test_dataset,
                &user_encoder,
                &item_encoder,
                &device,
            )?;
        }
        Commands::Search { query } => {
            println!("--- Search Mode ---");
            println!("Query: {}", query);
            let target_user_id = "1";
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &["model.safetensors"],
                    candle_core::DType::F32,
                    &device,
                )?
            };

            let user_encoder = IdEncoder::load("user_encoder.json")?;
            let item_encoder = IdEncoder::load("item_encoder.json")?;
            let model =
                CollaborativeFilteringModel::new(vb, user_encoder.len(), item_encoder.len(), 32)?;

            let candidates = search_movie(100, &query).await?;
            let candidate_ids: Vec<u32> = candidates
                .iter()
                .map(|point| {
                    match point
                        .id
                        .as_ref()
                        .unwrap()
                        .point_id_options
                        .as_ref()
                        .unwrap()
                    {
                        qdrant_client::qdrant::point_id::PointIdOptions::Num(num) => *num as u32,
                        _ => 0,
                    }
                })
                .collect();

            let id2title = read_movie("data/movielens_small/movies.csv")
                .expect("Movieファイルを読み込めませんでした");

            if let Some(user_idx) = user_encoder.encode(target_user_id) {
                println!("Generating recommendations for User: {}", target_user_id);

                // 全アイテムに対して予測を行うためのn湯力データを作成
                let n_items = candidates.len();

                let user_indices: Vec<u32> = vec![user_idx as u32; n_items];
                let item_indices: Vec<u32> = candidate_ids.clone();

                let user_input = Tensor::from_vec(user_indices, n_items, &device)?;
                let item_input = Tensor::from_vec(item_indices, n_items, &device)?;

                let scores = model.forward(&user_input, &item_input)?.to_vec1::<f32>()?;

                // スコアとアイテムIDをペアにソート
                // (index, score)の形にする
                let mut scored_items: Vec<(usize, f32)> =
                    scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();

                scored_items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                println!("--- Top 10 Recommended Movies ---");
                for (idx, score) in scored_items.iter().take(10) {
                    // 数値ID -> 元の映画ID
                    let real_ids = candidate_ids[*idx];
                    let original_id = item_encoder.decode(real_ids as usize).unwrap();

                    let title = id2title
                        .get(original_id)
                        .map(|s| s.as_str())
                        .unwrap_or("Unknown Title");
                    println!("Score: {:.4} | {}", score, title);
                }
            } else {
                println!("User {} not found in training data.", target_user_id);
            }
        }
        Commands::Server => {
            info!("--- Server Mode: Staring on 0.0.0.0:3000");

            let app_state = AppState::load(&args, &device)?;
            let shared_state = Arc::new(app_state);

            println!("Stateの読み込み完了");
            let app = Router::new()
                .route("/recommend", get(hybrid_recommendation))
                .with_state(shared_state);

            let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
            println!("serverの起動開始");
            axum::serve(listener, app).await?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Result;
    #[test]
    fn test_training_loop() -> Result<()> {
        let device = Device::Cpu;
        let user_ids = Tensor::new(&[0u32, 0u32], &device)?;
        let item_ids = Tensor::new(&[0u32, 1u32], &device)?;
        let ratings = Tensor::new(&[5.0f32, 1.0], &device)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = CollaborativeFilteringModel::new(vb, 1, 2, 4)?;

        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;

        for step in 0..10 {
            let ys = model.forward(&user_ids, &item_ids)?;
            let loss = mse(&ratings, &ys)?;
            println!("Step {step}: Loss = {}", loss.to_scalar::<f32>()?);
        }
        Ok(())
    }
}
