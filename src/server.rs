use crate::bert::BertEncoder;
use crate::cf_model::CollaborativeFilteringModel;
use crate::datasets::IdEncoder;
use crate::read_movie;
use axum::response::{IntoResponse, Response};
use axum::http::StatusCode;
use thiserror::Error;
use candle_nn::VarBuilder;
use qdrant_client::Qdrant;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::server::AppError::{Unexpected, UserNotFound};

pub struct AppState {
    pub ranking_model: std::sync::Arc<CollaborativeFilteringModel>,
    pub embedding_model: std::sync::Arc<BertEncoder>,
    pub qdrant_client: std::sync::Arc<Qdrant>,
    pub user_encoder: std::sync::Arc<IdEncoder>,
    pub item_encoder: std::sync::Arc<IdEncoder>,
    pub id2title: std::sync::Arc<HashMap<String, String>>,
}

impl AppState {
    pub fn load(arg: &crate::Args, device: &candle_core::Device) -> anyhow::Result<AppState> {
        let user_encoder = IdEncoder::load("user_encoder.json")?;
        let item_encoder = IdEncoder::load("item_encoder.json")?;

        let id2title = read_movie(&arg.movies_path).expect("Movieファイルを読み込めませんでした");

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &["model.safetensors"],
                candle_core::DType::F32,
                device,
            )?
        };

        let model =
            CollaborativeFilteringModel::new(vb, user_encoder.len(), item_encoder.len(), 32)?;

        let bert = BertEncoder::new(device)?;
        let qdrant_client = Qdrant::from_url(&arg.qdrant_url).build()?;

        Ok(AppState {
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
    pub limit: usize,
}

#[derive(Serialize)]
pub struct RecommendationResult {
    pub score: f64,
    pub title: String,
    pub item_id: String,
}

#[derive(Error, Debug)]
pub enum AppError {
    // 1. 特定したいエラー: ユーザーが見つからない
    #[error("User not found: {0}")]
    UserNotFound(String),

    // 2. その他の予期せぬエラー: Anyhowにラップして任せる
    #[error(transparent)]
    Unexpected(#[from] anyhow::Error)
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            // ユーザーが見つからない場合
            AppError::UserNotFound(user_id) => {
                (StatusCode::NOT_FOUND, format!("User {} not found", user_id))
            },
            // 予期せぬエラー (Qdrantが落ちている、バグ)
            AppError::Unexpected(err) => {
                tracing::error!("Internal Server Error: {:?}", err) ;
                (StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error".to_string())
            }
        };
        (status, message).into_response()

    }
}