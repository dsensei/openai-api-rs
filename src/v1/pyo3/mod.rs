use pyo3::{types::PyModule, FromPyObject, IntoPy, PyObject, Python};
use serde::{Deserialize, Serialize};


#[derive(FromPyObject, Debug, Serialize, Deserialize, Clone)]
pub struct LoraRequest {
    lora_id: String,
    lora_int_id: i32,
    lora_local_path: String,
}

impl IntoPy<PyObject> for LoraRequest {
    fn into_py(self, py: Python) -> PyObject {
        let py_module = PyModule::import(py, "vllm.lora.request").unwrap();
        let py_class = py_module.getattr("LoRARequest").unwrap();
        let lora_request = py_class
            .call1((self.lora_id, self.lora_int_id, self.lora_local_path))
            .unwrap();
        lora_request.into_py(py)
    }
}