pub fn calculate_precision(predictions: Vec<String>, labels: Vec<String>) -> f32 {
    if predictions.is_empty() {
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
#[cfg(test)]
mod tests {
    use crate::metrics::calculate_precision;
    #[test]
    fn test_precision() {
        let predictions = vec!["1".to_string(), "2".to_string(), "3".to_string()];
        let labels = vec!["3".to_string(), "5".to_string()];

        let actual = calculate_precision(predictions, labels);
        assert_eq!(actual, 1.0 / 3.0);
    }
    #[test]
    fn test_precision_when_prediction_is_nothing() {
        let predictions = vec![];
        let labels = vec!["3".to_string(), "5".to_string()];

        let actual = calculate_precision(predictions, labels);
        assert_eq!(actual, 0.0);
    }
}
