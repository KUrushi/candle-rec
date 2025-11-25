use std::io::stderr;
use candle_core::{Tensor, Result};
fn mse(labels: &Tensor, predictions: &Tensor) -> Result<Tensor> {
    (labels-predictions)?.sqr()?.mean_all()
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_mse() -> Result<()> {
        let labels = Tensor::new(&[1.0f32, 2.0, 3.0], &candle_core::Device::Cpu)?;
        let predictions = Tensor::new(&[1.0f32, 1.0, 2.0], &candle_core::Device::Cpu)?;

        let actual = mse(&labels, &predictions)?;
        let actual = actual.to_scalar::<f32>()?;
        let expected = candle_nn::loss::mse(&labels, &predictions)?.to_scalar::<f32>()?;

        assert!((actual - expected).abs() <= 0.0001);

        Ok(())
    }
}