#[derive(Debug, Clone)]
pub struct Candidate {
    pub item_id: String,
    pub score: f32
}

#[derive(Debug, Clone)]
pub struct Interaction {
    pub user_id: String,
    pub item_id: String,
    pub timestamp: usize
}