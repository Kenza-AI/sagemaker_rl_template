"""
ONNX Utils to support multiple output heads in agent networks, until future releases of MXNet support this.
"""
import onnx
from onnx import helper, checker, TensorProto

        
def make_output(node_name, shape):
    """
    Given a node name and output shape, will construct the correct Protobuf object.
    """
    return helper.make_tensor_value_info(
        name=node_name,
        elem_type=TensorProto.FLOAT,
        shape=shape
    )


def save_model(model, output_nodes, filepath):
    """
    Given an in memory model, will save to disk at given filepath.
    """
    new_graph = helper.make_graph(nodes=model.graph.node,
                                  name='new_graph',
                                  inputs=model.graph.input,
                                  outputs=output_nodes,
                                  initializer=model.graph.initializer)
    checker.check_graph(new_graph)
    new_model = helper.make_model(new_graph)
    with open(filepath, "wb") as file_handle:
        serialized = new_model.SerializeToString()
        file_handle.write(serialized)
 

def fix_onnx_model(filepath):
    """
    Applies an inplace fix to ONNX file from Coach. 
    """
    model = onnx.load_model(filepath)
    output_nodes = get_correct_outputs(model)
    save_model(model, output_nodes, filepath)
