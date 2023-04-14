#pragma once

#include "rep/call_graph_runner.h"
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
#include <environ/env_page.h>

namespace plearn::env {


	//NOTE: THIS IS ARCHIVED CODE. IT IS NOT USED ANYMORE.

	/*using diff_cache = tensor_p; */

	/*struct diff_cache_info { */
	/*	bool required{false}; */
	/*	shape_t shape; */
	/*}; */


	/*class fp_op_diff_env { */
	/*	public: */
	/*		fp_op_diff_env( */
	/*				unique_ptr<fw_op_diff_backend_t>&& diff_backend, */
	/*				vector<borrowed_ptr<grad_map>>&& in_grad_maps, */
	/*				borrowed_ptr<grad_map> out_grads */
	/*				) : */
	/*			diff_backend_{std::move(diff_backend)}, */ 
	/*			in_grad_maps_{std::move(in_grad_maps)}, out_grads_{out_grads} {} */

	/*		void execute( */
	/*				const vector<tensor_p>& inputs, */
	/*				const tensor_p& output */
	/*				) { */
	/*			diff_backend_->reset(inputs, output); */
	/*			for (unsigned idx = 0; idx < in_grad_maps_.size(); ++idx) { */
	/*				auto& in_grad_map = *in_grad_maps_[idx]; */
	/*				for (auto& [varn_id, var_in_grad]: in_grad_map) { */
	/*					if (var_in_grad.identity_) { */
	/*						//identity requires special handling */
	/*						diff_backend_->update_grad_with_identity(idx, out_grads_->at(varn_id).grad_); */
	/*						continue; */
	/*					} */
	/*					diff_backend_->update_grad(idx, var_in_grad.grad_, out_grads_->at(varn_id).grad_); */
	/*				} */
	/*			} */
	/*		} */

	/*	private: */
	/*		unique_ptr<fw_op_diff_backend_t> diff_backend_; */
			
	/*		/1* diff_cache cache_; *1/ */
	/*		const vector<borrowed_ptr<grad_map>> in_grad_maps_; */
	/*		borrowed_ptr<grad_map> out_grads_; */
	/*}; */



	/*class fp_diff_env : public diff_page { */
	/*	public: */
	/*		fp_diff_env( */
	/*				const call_graph& cg, */
	/*				borrowed_ptr<diff_info> fp_diff, */
	/*				borrowed_ptr<backend_t> backend */
	/*				) : */ 
	/*			cg_{cg}, fp_diff_{fp_diff}, backend_{backend} {} */

	/*		/1** */
	/*		 * Initializes the gradients_ map with variable nodes. */
	/*		 *1/ */
	/*		void init() { */
	/*			//insert variable nodes */
	/*			for (auto varn_id : fp_diff_->variable_nodes()) { */
	/*				grad_system_[varn_id] = {}; */
	/*				auto& var_shape = cg_.data_nodes_.at(varn_id).shape_; */
	/*				grad_system_[varn_id][varn_id] = {varn_id, varn_id, {var_shape, var_shape}, true}; */
	/*			} */
	/*			//insert flow and out nodes */
	/*			for (auto& [tens_id, _]: cg_.flow_nodes_) { */
	/*				if (std::ranges::count(cg_.in_nodes_, tens_id) > 0) */
	/*					continue; //skip input nodes */
	/*				grad_system_.insert({tens_id, {}}); */
	/*				auto& out_shape = cg_.flow_nodes_.at(tens_id).shape_; */
	/*				for (auto& var_id: fp_diff_->dependencies().at(tens_id).variable_dependencies()) { */
	/*					auto& var_shape = cg_.data_nodes_.at(var_id).shape_; */
	/*					auto grad_shape = var_shape*out_shape; */
	/*					auto back_t =backend_->create_tensor(grad_shape).release(); */
	/*					grad_system_[tens_id].insert({var_id, {var_id, tens_id, */ 
	/*						{var_shape, out_shape, shared_ptr<tensor_back_t>(back_t)}}}); */
	/*				} */
	/*			} */
	/*			//populate op_diff_envs_ */
	/*			for (auto& [opn_id, opn]: cg_.op_nodes_) { */
	/*				auto& op = opn.op_; */
	/*				vector<borrowed_ptr<grad_map>> in_grad_maps(opn.inputs_.size()); */
	/*				std::transform(opn.inputs_.begin(), opn.inputs_.end(), in_grad_maps.begin(), */
	/*						[this](auto in_id) { return &grad_system_[in_id]; }); */
	/*				auto& out_grad_map = grad_system_[opn.out_]; */
	/*				auto diff_backend = backend_->create_op_fw_diff_backend(op); */
	/*				op_diff_envs_[opn_id] = std::make_unique<fp_op_diff_env>( */
	/*						std::move(diff_backend), std::move(in_grad_maps), &out_grad_map); */
	/*			} */
	/*		} */

	/*		void reset() override { */
	/*			for (auto& [_, grad_map]: grad_system_) { */
	/*				for (auto& [__, node_grad]: grad_map) { */
	/*					if (node_grad.grad_.back_) */
	/*						node_grad.grad_.back_->zero(); */
	/*				} */
	/*			} */
	/*		} */

	/*		void calc_diffs(exec_page_tensors& tensors) override { */
	/*			call_graph_forward_runner runner{cg_}; */
	/*			runner.run([this, &tensors] (auto& opn) { */
	/*				vector<tensor_p> inputs(opn.inputs_.size()); */
	/*				std::transform(opn.inputs_.begin(), opn.inputs_.end(), inputs.begin(), */
	/*						[this, &tensors](auto in_id) { return tensors[in_id]; }); */
	/*				auto& output = tensors[opn.out_]; */
	/*				calc_diff(opn, inputs, output); */
	/*			}); */
	/*		} */

	/*		void calc_diff(const op_node& opn, */
	/*				const vector<tensor_p>& inputs, const tensor_p& output) { */
	/*			op_diff_envs_[opn.id_]->execute(inputs, output); */
	/*		} */

	/*	borrowed_ptr<grad_system> get_grad_system() override { return &grad_system_; } */

	/*	private: */
	/*		const call_graph& cg_; */
	/*		borrowed_ptr<diff_info> fp_diff_; */
	/*		borrowed_ptr<backend_t> backend_; */

	/*		unordered_map<op_node_id, unique_ptr<fp_op_diff_env>> op_diff_envs_; */
	/*		grad_system grad_system_; */
	/*}; */


	/*class fp_diff_env_builder { */

	/*}; */


}

