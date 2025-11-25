use std::collections::HashMap;
use std::path::PathBuf;
use candle_core::{DType, Device, Tensor};
use types::Interaction;
use candle_nn::{VarBuilder, VarMap, SGD, Optimizer};
use candle_nn::loss::mse;
use crate::cf_model::CollaborativeFilteringModel;
use crate::datasets::{split_data, DataLoader, IdEncoder, TensorDataset};
use crate::types::Movie;

mod types;
mod recommenders;
mod metrics;
mod cf_model;
mod loss;
mod datasets;

fn read_interactions(path: &str) -> Result<Vec<Interaction>, Box<dyn std::error::Error>> {
    let mut data = csv::Reader::from_path(PathBuf::from(path)).expect("ファイルパスが開けませんでした");
    let data = data.deserialize().map(|x|
        x.expect("行をパースできませんでした")).collect::<Vec<Interaction>>();
    Ok(data)
}

fn read_movie(path: &str) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    let mut data = csv::Reader::from_path(PathBuf::from(path)).expect("ファイルパスが開けませんでした");
    let mut mapper = HashMap::new();
    for result in data.deserialize() {
        let record: Movie = result?;
        mapper.insert(record.item_id, record.title);
    }
    Ok(mapper)
}

// fn train(user_ids: &Tensor,
//          item_ids: &Tensor,
//          ratings: &Tensor) -> candle_core::Result<()> {
//     let varmap = VarMap::new();
//     let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
//
//     let model = CollaborativeFilteringModel::new(
//         vb.clone(),
//         1,
//         1,
//         10,
//     )?;
//
//     let mut opt = SGD::new(varmap.all_vars(), 0.01)?;
//     for step in 0..100{
//         let ys = model.forward(user_ids, item_ids)?;
//         let loss = mse(&ys, ratings)?;
//         opt.backward_step(&loss)?
//     }
//     Ok(())
// }
fn main() -> anyhow::Result<()> {
    let path = "data/movielens_small/ratings.csv";
    let embedding_dim = 32;
    let learning_rate = 0.5;
    let batch_size = 64;
    let epochs = 50usize;
    let device = Device::Cpu;
    let lambda_ = 0.0001;

    println!("Loading data from {path}");

    let all_interactions = read_interactions(path).expect("データが読めませんでした");
    println!("Total interactions: {}", all_interactions.len());

    let (train_data, test_data) = split_data(all_interactions, 0.8);
    println!("Train: {}, Test: {}", train_data.len(), test_data.len());

    // Encoderの作成 (学習データに含まれるIDのみを知っている状態にする)
    let user_encoder = IdEncoder::new(train_data.iter().map(|x| &x.user_id));
    let item_encoder = IdEncoder::new(train_data.iter().map(|x| &x.item_id));

    println!("Users: {}, Items: {}", user_encoder.len(), item_encoder.len());

    // Datasetの作成
    let train_dataset = TensorDataset::new(&train_data, &user_encoder, &item_encoder, &device)?;
    let test_dataset = TensorDataset::new(&test_data, &user_encoder, &item_encoder, &device)?;


    // 3. モデル構築
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

    let model = CollaborativeFilteringModel::new(
        vb,
        user_encoder.len(),
        item_encoder.len(),
        embedding_dim,
    )?;

    let mut opt = SGD::new(varmap.all_vars(), learning_rate)?;

    println!("Start training...");


    for epoch in 1..=epochs {
        let mut total_train_loss = 0.0;
        let mut train_steps = 0;

        let train_loader = DataLoader::new(train_dataset.clone(), batch_size);
        for (u, i, r) in train_loader {
            let logits = model.forward(&u, &i)?;
            let mut loss = mse(&logits, &r)?;
            let mut l2_reg = Tensor::zeros((), DType::F32, &device)?;

            for var in varmap.all_vars() {
                l2_reg = (l2_reg + var.sqr()?.sum_all()?)?;
            }
            loss = (loss + (l2_reg * lambda_)?)?;

            opt.backward_step(&loss)?;
            total_train_loss += loss.to_scalar::<f32>()?;
            train_steps += 1;
        }

        let mut total_test_loss = 0.0;
        let mut test_steps = 0;

        let test_loader = DataLoader::new(test_dataset.clone(), batch_size);
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

    let id2title = read_movie("data/movielens_small/movies.csv").expect("Movieファイルを読み込めませんでした");
    let test_user_id = "1";

    if let Some(user_idx) = user_encoder.encode(test_user_id) {
        println!("Generating recommendations for User: {}", test_user_id);

        // 全アイテムに対して予測を行うためのn湯力データを作成
        let n_items = item_encoder.len();

        let user_indices: Vec<u32> = vec![user_idx as u32; n_items];
        let item_indices: Vec<u32> = (0..n_items as u32).collect();

        let user_input = Tensor::from_vec(user_indices, n_items, &device)?;
        let item_input = Tensor::from_vec(item_indices, n_items, &device)?;

        let scores = model.forward(&user_input, &item_input)?.to_vec1::<f32>()?;

        // スコアとアイテムIDをペアにソート
        // (index, score)の形にする
        let mut scored_items: Vec<(usize, f32)> = scores.iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();

        scored_items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("--- Top 10 Recommended Movies ---");
        for (idx, score) in scored_items.iter().take(10) {
            // 数値ID -> 元の映画ID
            let original_id = item_encoder.decode(*idx).unwrap();
            let title = id2title.get(original_id).map(|s| s.as_str()).unwrap_or("Unknown Title");
            println!("Score: {:.4} | {}", score, title);
        }
    } else {
        println!("User {} not found in training data.", test_user_id);
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

