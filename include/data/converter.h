#pragma once

#include <cstdint>

#include <optional>
#include <rep/call_graph.h>
#include <data/gen/call_graph.pb.h>

namespace plearn::data {

	using namespace plearn::rep;


	template <typename T, typename S>
		vector<T> to_vector(const google::protobuf::RepeatedField<S>& repeated_field) {
			return vector<T>(repeated_field.begin(), repeated_field.end());
		}
	
	template <typename T>
		vector<T> to_vector(const google::protobuf::RepeatedField<T>& repeated_field) {
			return vector<T>(repeated_field.begin(), repeated_field.end());
		}

	class ShapeConverter {
		public:
			static shape_t to_shape(const ShapeM& s) {
				auto dims = to_vector<uint64_t, long>(s.dims());
				return shape_t(dims);
			}

			static ShapeM* to_proto(const shape_t& s) {
				ShapeM* s_m = new ShapeM;
				s_m->set_rank(s.rank);
				for (auto dim : s.dims) {
					s_m->add_dims(dim);
				}
				return s_m;
			}
	};

	class OpTypeConverter {
		public:
			static op_type to_op_type(OpTypeM type) {
				return (op_type) type;
			}

			static OpTypeM to_proto(op_type type) {
				return (OpTypeM) type;
			}
	};

	class OperationConverter {
		public:
			static operation to_operation(const OperationM& op) {
				return {.type_=OpTypeConverter::to_op_type(op.optype())};
			}

			static OperationM* to_proto(const operation& op) {
				OperationM* op_m = new OperationM;
				op_m->set_optype(OpTypeConverter::to_proto(op.type_));
				return op_m;
			}
	};

	class OpNodeConverter {
		public:
			static op_node to_op_node(const OpNodeM& node) {
				return op_node{
					.id_=node.id(),
					.op_=OperationConverter::to_operation(node.op()),
					.inputs_=to_vector(node.inputs()),
					.out_=node.output()
				};
			}

			static OpNodeM* to_proto(const op_node& node) {
				OpNodeM* node_m = new OpNodeM;
				node_m->set_id(node.id_);
				node_m->set_allocated_op(OperationConverter::to_proto(node.op_));
				for (auto input: node.inputs_) {
					node_m->add_inputs(input);
				}
				node_m->set_output(node.out_);
				return node_m;
			}
	};

	class TensorNodeConverter {
		public:
			static tensor_node to_tensor_node(const TensorNodeM& node) {
				auto input = node.has_input() ? std::optional(node.input()) : std::nullopt;
				return tensor_node{
					.id_=node.id(),
					.shape_= ShapeConverter::to_shape(node.shape()),
					.input_{input},
					.outputs_{to_vector(node.outputs())}
				};
			}

			static TensorNodeM* to_proto(const tensor_node& node) {
				TensorNodeM* node_m = new TensorNodeM;
				node_m->set_id(node.id_);
				node_m->set_allocated_shape(ShapeConverter::to_proto(node.shape_));
				if (node.input_.has_value()) {
					node_m->set_input(node.input_.value());
				}
				for (auto output: node.outputs_) {
					node_m->add_outputs(output);
				}
				return node_m;
			}
	};

	class GraphConverter {
		public:
			static call_graph to_call_graph(const CallGraphM& graph) {
				call_graph cg;
				for (auto& [id, node]: graph.datanodes()) {
					cg.data_nodes_[id] = TensorNodeConverter::to_tensor_node(node);
				}
				for (auto& [id, node]: graph.flownodes()) {
					cg.flow_nodes_[id] = TensorNodeConverter::to_tensor_node(node);
				}
				for (auto& [id, node]: graph.opnodes()) {
					cg.op_nodes_[id] = OpNodeConverter::to_op_node(node);
				}
				for (auto inn: graph.innodes()) {
					cg.in_nodes_.push_back(inn);
				}
				for (auto outn: graph.outnodes()) {
					cg.out_nodes_.push_back(outn);
				}
				for (auto internaln: graph.internalnodes()) {
					cg.internal_nodes_.push_back(internaln);
				}
				return cg;
			}

			static CallGraphM* to_proto(const call_graph& graph) {
				CallGraphM* graph_m = new CallGraphM;
				for (auto& [id, node]: graph.data_nodes_) {
					//TODO dont allocate on heap
					TensorNodeM* node_m = TensorNodeConverter::to_proto(node);
					(*graph_m->mutable_datanodes())[id] = *node_m;
					delete node_m;
				}
				for (auto& [id, node]: graph.flow_nodes_) {
					TensorNodeM* node_m = TensorNodeConverter::to_proto(node);
					(*graph_m->mutable_flownodes())[id] = *node_m;
					delete node_m;
				}
				for (auto& [id, node]: graph.op_nodes_) {
					OpNodeM* node_m = OpNodeConverter::to_proto(node);
					(*graph_m->mutable_opnodes())[id] = *node_m;
					delete node_m;
				}
				for (auto id: graph.in_nodes_) {
					graph_m->add_innodes(id);
				}
				for (auto id: graph.out_nodes_) {
					graph_m->add_outnodes(id);
				}
				for (auto id: graph.internal_nodes_) {
					graph_m->add_internalnodes(id);
				}
				return graph_m;
			}
	};


}
