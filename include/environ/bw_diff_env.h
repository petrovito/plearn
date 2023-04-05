#pragma once

#include <algorithm>
#include <bits/ranges_algo.h>
#include <cassert>
#include <cstdint>
#include <memory>
#include <ranges>
#include <unordered_map>
#include <vector>

#include <rep/rep_types.h>
#include <rep/call_graph.h>
#include <rep/diff_info.h>
#include <environ/env_types.h>
#include <environ/exec_env.h>

namespace plearn::env {

	class bw_op_diff_env {
		public:
			bw_op_diff_env(
					unique_ptr<bw_op_diff_backend_t>&& diff_backend,
					read_ptr<grad_map> out_grad_map,
					vector<borrowed_ptr<grad_map>>&& in_grad_maps
					) :
				diff_backend_(std::move(diff_backend)),
				out_grad_map_(out_grad_map),
				in_grad_maps_(std::move(in_grad_maps)) {}

			void execute(
					const vector<tensor_p>& inputs,
					const tensor_p& output
					) {
				diff_backend_->reset(inputs, output);
				for (unsigned in_idx = 0; in_idx < in_grad_maps_.size(); ++in_idx) {
					auto in_grad_map = in_grad_maps_[in_idx];
					for (auto& [outn_id, in_outn_grad] : *in_grad_map) {
						if (!out_grad_map_->contains(outn_id)) continue;
						auto& out_outn_grad = out_grad_map_->at(outn_id);
						diff_backend_->update_grad(in_idx, out_outn_grad.grad_, in_outn_grad.grad_);
					}
				}
			}
		private:
			unique_ptr<bw_op_diff_backend_t> diff_backend_;

			read_ptr<grad_map> out_grad_map_;
			vector<borrowed_ptr<grad_map>> in_grad_maps_;
	};


	class bw_diff_env : public diff_env {
		public:
			bw_diff_env(
					const call_graph& cg,
					borrowed_ptr<diff_info> diff_info,
					borrowed_ptr<backend_t> backend,
					unordered_map<op_node_id, unique_ptr<bw_op_diff_env>>&& op_diff_envs,
					grad_system&& grad_system
					) :
				cg_(cg), diff_info_(diff_info), backend_(backend),
				op_diff_envs_(std::move(op_diff_envs)), grad_system_(std::move(grad_system)) {}

			void reset() override {
				for (auto& [nid, grad_map] : grad_system_) {
					//dont reset outnodes
					if (std::ranges::count(cg_.out_nodes_, nid)) continue;
					for (auto& [_, grad] : grad_map)
						if (grad.grad_.back_)
							grad.grad_.back_->zero();
				}
			}

			void calc_diff(const op_node& opn, 
					const vector<tensor_p>& inputs, const tensor_p& output) override {
				op_diff_envs_[opn.id_]->execute(inputs, output);
			}

			void calc_diffs(section_exec_tensors& tensors) override {
				call_graph_backward_runner runner{cg_};
				runner.run([this, &tensors] (auto& opn) {
					vector<tensor_p> inputs(opn.inputs_.size());
					std::transform(opn.inputs_.begin(), opn.inputs_.end(), inputs.begin(),
							[this, &tensors](auto in_id) { return tensors[in_id]; });
					auto& output = tensors[opn.out_];
					calc_diff(opn, inputs, output);
				});
			}

			borrowed_ptr<grad_system> get_grad_system() override { return &grad_system_; }

		private:
			const call_graph& cg_;
			borrowed_ptr<diff_info> diff_info_;
			borrowed_ptr<backend_t> backend_;

			unordered_map<op_node_id, unique_ptr<bw_op_diff_env>> op_diff_envs_;
			grad_system grad_system_;
	};


	class bw_diff_env_builder {
		public:
			bw_diff_env_builder(
					const call_graph& cg,
					borrowed_ptr<diff_info> diff_info,
					borrowed_ptr<backend_t> backend
					) :
				cg_(cg), diff_info_(diff_info), backend_(backend) {}


			bw_diff_env_builder& allocate_grad_tensors() {
				//insert output nodes as identity
				for (auto& outn_id: cg_.out_nodes_) {
					auto& outn = cg_.flow_nodes_.at(outn_id);
					auto grad_tens_shape = outn.shape_ * outn.shape_;
					auto back_tens = backend_->create_tensor(grad_tens_shape, tensor_init::identity).release();
					grad_system_[outn_id][outn_id] = {outn_id, outn_id, 
						{outn.shape_, outn.shape_, shared_ptr<tensor_back_t>(back_tens)}, true};
				}
				//insert flow and data nodes
				//flow nodes
				for (auto& [flown_id, flown]: cg_.flow_nodes_) {
					if (std::ranges::count(cg_.out_nodes_, flown_id)) continue;
					auto& deps = diff_info_->dependencies().at(flown_id);
					//if flown doesnt depend on any variable, skip
					if (!deps.depends_on_any()) continue;
					for (auto& outn_id: cg_.out_nodes_) {
						//only if outn depends on flown
						if (!deps.output_dependant(outn_id)) continue;
						auto& outn = cg_.flow_nodes_.at(outn_id);
						auto grad_tens_shape = flown.shape_ * outn.shape_;
						auto back_tens = backend_->create_tensor(grad_tens_shape).release();
						grad_system_[flown_id][outn_id] = {flown_id, outn_id, 
							{flown.shape_, outn.shape_, shared_ptr<tensor_back_t>(back_tens)}, false};
					}
				}
				//data nodes
				for (auto& [datan_id, datan]: cg_.data_nodes_) {
					auto& deps = diff_info_->dependencies().at(datan_id);
					//if flown doesnt depend on any variable, skip
					if (!deps.depends_on_any()) continue;
					for (auto& outn_id: cg_.out_nodes_) {
						//only if outn depends on flown
						if (!deps.output_dependant(outn_id)) continue;
						auto& outn = cg_.flow_nodes_.at(outn_id);
						auto grad_tens_shape = datan.shape_ * outn.shape_;
						auto back_tens = backend_->create_tensor(grad_tens_shape).release();
						grad_system_[datan_id][outn_id] = {datan_id, outn_id, 
							{datan.shape_, outn.shape_, shared_ptr<tensor_back_t>(back_tens)}, false};
					}
				}
				//populate op_diff_envs_
				for (auto& [opn_id, opn]: cg_.op_nodes_) {
					auto& op = opn.op_;
					auto& out_grad_map = grad_system_.at(opn.out_);
					vector<borrowed_ptr<grad_map>> in_grad_maps(opn.inputs_.size());
					std::transform(opn.inputs_.begin(), opn.inputs_.end(), in_grad_maps.begin(),
							[this](auto in_id) { return &grad_system_[in_id]; });
					auto op_diff_backend = backend_->create_op_bw_diff_backend(op);
					op_diff_envs_[opn_id] = std::make_unique<bw_op_diff_env>(
							std::move(op_diff_backend), &out_grad_map, std::move(in_grad_maps));
				}
				return *this;
			}
			unique_ptr<bw_diff_env> build() {
				return std::make_unique<bw_diff_env>(
						cg_, diff_info_, backend_, 
						std::move(op_diff_envs_), std::move(grad_system_));
			}

		private:
			const call_graph& cg_;
			borrowed_ptr<diff_info> diff_info_;
			borrowed_ptr<backend_t> backend_;

			unordered_map<op_node_id, unique_ptr<bw_op_diff_env>> op_diff_envs_;
			grad_system grad_system_;
	};

}

