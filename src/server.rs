use std::collections::HashMap;
use candle_core::Device;
use candle_nn::VarBuilder;
use qdrant_client::Qdrant;
use serde::{Deserialize, Serialize};
use crate::bert::BertEncoder;
use crate::cf_model::CollaborativeFilteringModel;
use crate::datasets::IdEncoder;
use crate::read_movie;

pub struct AppState {
    pub ranking_model: std::sync::Arc<CollaborativeFilteringModel>,
    pub embedding_model: std::sync::Arc<BertEncoder>,
    pub qdrant_client: std::sync::Arc<Qdrant>,
    pub user_encoder: std::sync::Arc<IdEncoder>,
    pub item_encoder: std::sync::Arc<IdEncoder>,
    pub id2title: std::sync::Arc<HashMap<String,String>>,
}

impl AppState {
    pub fn load(arg: &crate::Args, device: &candle_core::Device) -> anyhow::Result<AppState> {
        let user_encoder = IdEncoder::load("user_encoder.json")?;
        let item_encoder = IdEncoder::load("item_encoder.json")?;

        let id2title = read_movie(&arg.movies_path).expect("Movieファイルを読み込めませんでした");

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&["model.safetensors"], candle_core::DType::F32, device)?
        };

        let model = CollaborativeFilteringModel::new(
            vb,
            user_encoder.len(),
            item_encoder.len(),
            32
        )?;

        let bert = BertEncoder::new(device)?;
        let qdrant_client = Qdrant::from_url(&arg.qdrant_url).build()?;

        Ok(AppState{
            ranking_model: std::sync::Arc::new(model),
            embedding_model: std::sync::Arc::new(bert),
            qdrant_client: std::sync::Arc::new(qdrant_client),
            user_encoder: std::sync::Arc::new(user_encoder),
            item_encoder: std::sync::Arc::new(item_encoder),
            id2title: std::sync::Arc::new(id2title),
        })
    }
}
#[derive(Deserialize)]
pub struct RecommendQuery {
    pub user_id: String,
    pub query: String,
    pub limit: usize
}

#[derive(Serialize)]
pub struct RecommendationResult {
    pub score: f64,
    pub title: String,
    pub item_id: String
}