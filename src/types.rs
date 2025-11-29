use serde::{Deserialize, Serialize};
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Candidate {
    pub item_id: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interaction {
    #[serde(rename = "userId")]
    pub user_id: String,
    #[serde(rename = "movieId")]
    pub item_id: String,
    pub timestamp: usize,
    pub rating: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Movie {
    #[serde(rename = "movieId")]
    pub item_id: String,
    pub title: String,
    pub genres: String,
}
