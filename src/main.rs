use std::io::Error;

#[derive(Debug, Clone)]
struct Candidate {
    item_id: String,
    score: f32
}
trait Recommender {
    fn recommend(&self, user_id: String) -> Result<Vec<Candidate>, std::io::Error>;
}
struct RandomRecommender;

impl Recommender for RandomRecommender {
    fn recommend(&self, user_id: String) -> Result<Vec<Candidate>, Error> {
        Ok(vec![Candidate{item_id: "1".to_string(),
        score:0.1},
        Candidate{item_id: "2".to_string(),score:0.2},
        Candidate{item_id: "3".to_string(), score:0.3}])
    }
}

fn calculate_precision(predictions: Vec<String>, labels: Vec<String>) -> f32 {
    if predictions.len() == 0 {
        return 0.0;
    }
    let mut correct = 0_usize;
    for prediction in &predictions {
        if labels.contains(prediction) {
            correct += 1;
        }
    }
    correct as f32 / predictions.len() as f32
}




fn main() {
    let rec = RandomRecommender{};
    println!("{:?}", rec.recommend("1".to_string()).unwrap());
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn test_precision() {
        let predictions = vec!["1".to_string(), "2".to_string(), "3".to_string()];
        let labels = vec!["3".to_string(), "5".to_string()];

        let actual = calculate_precision(predictions, labels);
        assert_eq!(actual, 1.0/3.0);
    }
    #[test]
    fn test_precision_when_prediction_is_nothing() {
        let predictions = vec![];
        let labels = vec!["3".to_string(), "5".to_string()];

        let actual = calculate_precision(predictions, labels);
        assert_eq!(actual, 0.0);
    }
}