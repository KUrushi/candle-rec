use crate::types::{Candidate, Interaction};
use std::collections::HashMap;
use std::io::Error;

pub trait Recommender {
    fn recommend(&self, user_id: String) -> Result<Vec<Candidate>, std::io::Error>;
}

pub struct RandomRecommender;

impl Recommender for RandomRecommender {
    fn recommend(&self, _user_id: String) -> Result<Vec<Candidate>, Error> {
        Ok(vec![
            Candidate {
                item_id: "1".to_string(),
                score: 0.1,
            },
            Candidate {
                item_id: "2".to_string(),
                score: 0.2,
            },
            Candidate {
                item_id: "3".to_string(),
                score: 0.3,
            },
        ])
    }
}

pub struct MostPopularRecommender {
    pub interactions: Vec<Interaction>,
}

impl Recommender for MostPopularRecommender {
    fn recommend(&self, _user_id: String) -> Result<Vec<Candidate>, Error> {
        let mut counter: HashMap<String, usize> = HashMap::new();
        for interaction in &self.interactions {
            *counter.entry(interaction.item_id.clone()).or_insert(0) += 1;
        }

        let mut results = Vec::new();
        for (key, value) in counter.iter() {
            results.push(Candidate {
                item_id: key.clone(),
                score: *value as f32,
            })
        }
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_ranking_recommender() {
        let recommender = MostPopularRecommender {
            interactions: vec![
                Interaction {
                    user_id: "1".to_string(),
                    item_id: "1".to_string(),
                    timestamp: 0,
                    rating: 1.0,
                },
                Interaction {
                    user_id: "1".to_string(),
                    item_id: "1".to_string(),
                    timestamp: 0,
                    rating: 1.0,
                },
                Interaction {
                    user_id: "1".to_string(),
                    item_id: "2".to_string(),
                    timestamp: 0,
                    rating: 1.0,
                },
            ],
        };
        let actual = recommender.recommend("1".to_string()).unwrap();
        let actual: Vec<String> = actual.iter().map(|x| x.item_id.clone()).collect();
        let expected = vec!["1".to_string(), "2".to_string()];

        assert_eq!(actual, expected);
    }
}
