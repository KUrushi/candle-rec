use recommenders::{MostPopularRecommender, Recommender};
use types::Interaction;
mod types;
mod recommenders;
mod metrics;

fn main() {
    let recommender = MostPopularRecommender{
        interactions: vec![
            Interaction{
                user_id:"1".to_string(),
                item_id:"1".to_string(),
                timestamp:0,
            },
            Interaction{
                user_id:"1".to_string(),
                item_id:"1".to_string(),
                timestamp:0,
            },
            Interaction{
                user_id:"1".to_string(),
                item_id:"2".to_string(),
                timestamp:0,
            },
            Interaction{
                user_id:"1".to_string(),
                item_id:"3".to_string(),
                timestamp:0,
            }
        ]
    };
    println!("{:?}", recommender.recommend("1".to_string()).unwrap());
}

