
use ort::session::{ SessionOutputs };
use ort::value::{ TensorValueType, Value };
use ort::tensor::{ Shape, PrimitiveTensorElementType };
use ndarray::{ ArrayD };
use std::fmt::Debug;


pub trait TensorProcessor{
    fn array_to_tensor<T>(array_in: ArrayD<T>) -> Value<TensorValueType<T>>
    where
        T: PrimitiveTensorElementType + Debug + Clone + 'static
    {
        Value::from_array(array_in).expect("Error creating out in tensorref")
    }

    fn infered_tensor<'a, T>(out: &'a SessionOutputs, key_name: &str) -> (&'a Shape, &'a [T])
    where 
        T: PrimitiveTensorElementType
    {
        let values = &out[key_name];
        values
            .try_extract_tensor::<T>()
            .expect(format!("Error extracting out tensor with key {}", key_name).as_str())
    }

    fn infered_to_array<'a, T>(infered_tensor: (&'a Shape, &'a [T])) -> ArrayD<T>
    where
        T: Clone
    {
        let shape: Vec<usize> = infered_tensor.0.iter().map(
            |&x| x as usize
        ).collect();
        ArrayD::<T>::from_shape_vec(
            shape, infered_tensor.1.to_vec()
        ).expect("Error in infered to array")
    }

    fn try_tensor<'a, T>(out: &'a SessionOutputs, key_name: &str) -> ArrayD<T>
    where 
        T: PrimitiveTensorElementType + Clone
    {
        Self::infered_to_array(Self::infered_tensor(out, key_name))
    }
}
