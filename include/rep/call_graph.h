#pragma once

#include <optional>
#include <rep/rep_types.h>

namespace plearn::rep {

	struct tensor_node {
		node_id id_;
		shape_t shape_;
		std::optional<node_id> input_{};
		vector<node_id> outputs_{};
		
		friend auto operator<=>(const tensor_node&, const tensor_node&) = default;
	};

	struct op_node {
		node_id id_;
		operation op_;

		vector<node_id> inputs_; //dependencies
		node_id out_{};

		static op_node identity(tensor_node_id id) { 
			return {-1, {op_type::identity}, {id}, id};
		}  

		friend auto operator<=>(const op_node&, const op_node&) = default;
	};



	class call_graph {
		public:
			/**
			 * flow_nodes are the union of input, output and internal nodes
			 * Where data flows through the graph.
			 */
			hash_map<node_id, tensor_node> flow_nodes_;
			/**
			 * Data nodes are the "parameters of the graph", the ones that are not
			 * influenced by the data flowing through the graph.
			 * AKA the trainable parameters.
			 */
			hash_map<node_id, tensor_node> data_nodes_;
			hash_map<node_id, op_node> op_nodes_;

			vector<node_id> in_nodes_;
			vector<node_id> out_nodes_;
			vector<node_id> internal_nodes_;

			friend bool operator==(const call_graph& a, const call_graph& b) = default;	
	};


	class call_graph_builder {
		public:
			node_id add_input_node(shape_t s) {
				return add_flow_node(s, std::nullopt);
			}

			node_id add_data_node(shape_t s) {
				node_id id = next_id_++;
				tensor_node tn{id, s};
				data_nodes_[id] = tn;
				tensor_nodes_[id] = &data_nodes_[id];
				return id;
			}
			
			//returns the id of the op node and the id of the output tensor node
			std::tuple<node_id, node_id> add_op_node(operation op,
					vector<node_id> inputs, shape_t out_shape) {
				node_id id = next_id_++;
				auto out_id = add_flow_node(out_shape, id);
				op_node on{id, op, inputs, out_id};
				op_nodes_[id] = on;
				//add op to tensor node outputs
				for (auto node_id: inputs) {
					tensor_nodes_[node_id]->outputs_.push_back(id);
				}
				return {id, out_id};
			}

			void make_output(node_id id) {
				//remove from internal and add to out
				auto it = std::find(internal_nodes_.begin(), internal_nodes_.end(), id);
				if (it != internal_nodes_.end()) {
					internal_nodes_.erase(it);
					out_nodes_.push_back(id);
				} else {
					throw std::runtime_error("node is not internal");
				}
			}

			call_graph build() {
				return {
					std::move(flow_nodes_),
					std::move(data_nodes_),
					std::move(op_nodes_),
					std::move(in_nodes_),
					std::move(out_nodes_),
					std::move(internal_nodes_)
				};
			}

		private:
			node_id add_flow_node(shape_t s, std::optional<node_id> input) {
				node_id id = next_id_++;
				tensor_node tn{id, s, input};
				flow_nodes_[id] = tn;
				tensor_nodes_[id] = &flow_nodes_[id];
				if (input.has_value()) {
					internal_nodes_.push_back(id);
				} else { //input node
					in_nodes_.push_back(id);
				}
				return id;
			}

			hash_map<node_id, tensor_node*> tensor_nodes_;
			hash_map<node_id, tensor_node> flow_nodes_;
			hash_map<node_id, tensor_node> data_nodes_;
			hash_map<node_id, op_node> op_nodes_;
			vector<node_id> in_nodes_;
			vector<node_id> out_nodes_;
			vector<node_id> internal_nodes_;
			node_id next_id_ = 0;

	};


}
