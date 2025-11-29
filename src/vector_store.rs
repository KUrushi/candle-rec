use anyhow::Result;
use candle_core::Tensor;
use qdrant_client::Payload;
use qdrant_client::Qdrant;
use std::collections::HashMap; // Clientの名前が変わっている場合があるので注意
// もし qdrant-client 1.16系なら Qdrant ではなく QdrantClient かもしれません。
// エラーが出なければそのままでOKです。
use crate::datasets::IdEncoder;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{
    CreateCollection, Distance, PointStruct, ScoredPoint, SearchPointsBuilder, UpsertPointsBuilder,
    VectorParams, VectorsConfig,
};
use serde_json::json;

pub const COLLECTION_NAME: &str = "movies";

pub async fn init_qdrant(url: &str) -> Result<Qdrant> {
    let client = Qdrant::from_url(url)
        .build()
        .expect("Qdrant was not created");

    if client.collection_exists(COLLECTION_NAME).await? {
        println!("Collection '{}' exists. Deleting...", COLLECTION_NAME);
        client.delete_collection(COLLECTION_NAME).await?;
    }

    println!("Creating collection '{}'...", COLLECTION_NAME);

    // 【修正点】 & を削除しました
    let result = client
        .create_collection(CreateCollection {
            collection_name: COLLECTION_NAME.to_string(),
            vectors_config: Some(VectorsConfig {
                config: Some(Config::Params(VectorParams {
                    size: 384,
                    // 【推奨】マジックナンバー '1' の代わりに Enum を使用
                    distance: Distance::Cosine.into(),
                    ..Default::default()
                })),
            }),
            ..Default::default()
        })
        .await?; // unwrap() ではなく ? を使うとエラーハンドリングが綺麗になります

    Ok(client)
}

pub async fn upsert_movies(
    client: &Qdrant,
    item_encoder: &IdEncoder,
    id2title: &HashMap<String, String>,
    embeddings: &Tensor,
) -> Result<()> {
    let n_items = item_encoder.len();
    let mut points = Vec::new();

    let vectors = embeddings.to_vec2::<f32>()?;

    println!("Preparing {} points for Qdrant...", n_items);

    for i in 0..n_items {
        let item_id_str = item_encoder.decode(i).unwrap_or("Unknown");
        let title = id2title
            .get(item_id_str)
            .cloned()
            .unwrap_or_else(|| "Unknown".to_string());
        let payload = Payload::try_from(json!({
            "oritinal_id": item_id_str,
            "title":title
        }))?;

        let point = PointStruct::new(i as u64, vectors[i].clone(), payload);

        points.push(point)
    }

    println!("Uploading points to Qdrant...");
    client
        .upsert_points(UpsertPointsBuilder::new(COLLECTION_NAME, points).wait(true))
        .await?;

    Ok(())
}

pub async fn search_movies(
    client: &Qdrant,
    query_vector: Vec<f32>,
    limit: u64,
) -> Result<Vec<ScoredPoint>> {
    println!("Searching ...");

    let search_result = client
        .search_points(
            SearchPointsBuilder::new(COLLECTION_NAME, query_vector, limit).with_payload(true),
        )
        .await
        .unwrap();

    println!("Found {} results:", search_result.result.len());

    let result = search_result.result;
    for point in result.iter() {
        let title = point
            .payload
            .get("title")
            .and_then(|v| v.kind.as_ref())
            .and_then(|k| match k {
                qdrant_client::qdrant::value::Kind::StringValue(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap_or("Unknown Title");

        println!("Score: {:.4} | Title: {}", point.score, title);
    }

    Ok(result)
}
