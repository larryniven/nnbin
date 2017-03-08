#include "ebt/ebt.h"
#include "speech/speech.h"
#include "nn/tensor-tree.h"
#include "nn/rsg.h"
#include "nn/nn.h"
#include <random>
#include <algorithm>

std::vector<std::shared_ptr<autodiff::op_t>> reconstruct(
    autodiff::computation_graph& comp_graph,
    std::vector<std::shared_ptr<autodiff::op_t>> const& seg_frames,
    std::shared_ptr<tensor_tree::vertex> var_tree,
    std::string const& label,
    std::unordered_map<std::string, int> const& label_id,
    int layer);

std::string load_label_batch(std::ifstream& ifs);

struct learning_env {

    speech::batch_indices frame_batch;
    speech::batch_indices label_batch;

    std::shared_ptr<tensor_tree::vertex> param;

    int layer;

    std::shared_ptr<tensor_tree::optimizer> opt;

    std::string output_param;
    std::string output_opt_data;

    std::vector<int> sample_indices;

    double step_size;
    double decay;
    double clip;

    double dropout;

    std::default_random_engine gen;
    int seed;

    std::vector<std::string> id_label;
    std::unordered_map<std::string, int> label_id;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> const& args);

    void run();

};

learning_env::learning_env(std::unordered_map<std::string, std::string> const& args)
    : args(args)
{
    frame_batch.open(args.at("frame-batch"));
    label_batch.open(args.at("label-batch"));

    std::string line;
    std::ifstream param_ifs { args.at("param") };
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    param = rsg::make_tensor_tree(layer);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    if (ebt::in(std::string("output-param"), args)) {
        output_param = args.at("output-param");
    }

    if (ebt::in(std::string("output-opt-data"), args)) {
        output_opt_data = args.at("output-opt-data");
    }

    if (ebt::in(std::string("step-size"), args)) {
        step_size = std::stod(args.at("step-size"));
    }

    decay = 0;
    if (ebt::in(std::string("decay"), args)) {
        decay = std::stod(args.at("decay"));
    }

    clip = 0;
    if (ebt::in(std::string("clip"), args)) {
        clip = std::stod(args.at("clip"));
    }

    dropout = 0;
    if (ebt::in(std::string("dropout"), args)) {
        dropout = std::stod(args.at("dropout"));
    }

    seed = 0;
    if (ebt::in(std::string("seed"), args)) {
        seed = std::stoi(args.at("seed"));
    }

    id_label = speech::load_label_set(args.at("label"));
    for (int i = 0; i < id_label.size(); ++i) {
        label_id[id_label[i]] = i;
    }

    if (ebt::in(std::string("opt-data"), args)) {
        if (ebt::in(std::string("decay"), args)) {
            opt = std::make_shared<tensor_tree::rmsprop_opt>(
                tensor_tree::rmsprop_opt(param, decay, step_size));
        } else if (ebt::in(std::string("const-step-update"), args)) {
            opt = std::make_shared<tensor_tree::const_step_opt>(
                tensor_tree::const_step_opt(param, step_size));
        } else {
            opt = std::make_shared<tensor_tree::adagrad_opt>(
                tensor_tree::adagrad_opt(param, step_size));
        }
    }

    if (ebt::in(std::string("opt-data"), args)) {
        std::string line;
        std::ifstream opt_data_ifs { args.at("opt-data") };
        std::getline(opt_data_ifs, line);
        opt->load_opt_data(opt_data_ifs);
        opt_data_ifs.close();
    }

    gen = std::default_random_engine { seed };

    sample_indices.resize(frame_batch.pos.size());

    for (int i = 0; i < sample_indices.size(); ++i) {
        sample_indices[i] = i;
    }

    if (ebt::in(std::string("shuffle"), args)) {
        std::shuffle(sample_indices.begin(), sample_indices.end(), gen);

        std::vector<unsigned long> pos = frame_batch.pos;
        for (int i = 0; i < sample_indices.size(); ++i) {
            frame_batch.pos[i] = pos[sample_indices[i]];
        }

        pos = label_batch.pos;
        for (int i = 0; i < sample_indices.size(); ++i) {
            label_batch.pos[i] = pos[sample_indices[i]];
        }
    }
}

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "rsg-unsup-learn",
        "Train a Recurrent Sequence Generator",
        {
            {"frame-batch", "", true},
            {"label-batch", "", true},
            {"param", "", true},
            {"opt-data", "", false},
            {"step-size", "", false},
            {"decay", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"clip", "", false},
            {"label", "", true},
            {"dropout", "", false},
            {"seed", "", false},
            {"shuffle", "", false},
            {"const-step-update", "", false},
        }
    };

    if (argc == 1) {
        ebt::usage(spec);
        exit(1);
    }

    auto args = ebt::parse_args(argc, argv, spec);

    for (int i = 0; i < argc; ++i) {
        std::cout << argv[i] << " ";
    }
    std::cout << std::endl;

    learning_env env { args };

    env.run();

    return 0;
}

void learning_env::run()
{
    int nsample = 0;

    while (nsample < frame_batch.pos.size()) {
        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch.at(nsample));
        std::string label = load_label_batch(label_batch.at(nsample));

        std::cout << "sample: " << sample_indices.at(nsample) << std::endl;
        std::cout << "frames: " << frames.size() << std::endl;
        std::cout << "label: " << label << std::endl;

        autodiff::computation_graph comp_graph;

        auto var_tree = tensor_tree::make_var_tree(comp_graph, param);

        std::vector<std::shared_ptr<autodiff::op_t>> seg_frames;

        for (int i = 0; i < frames.size() - 1; ++i) {
            seg_frames.push_back(comp_graph.var(
                la::tensor<double>(la::vector<double>(frames.at(i)))));
        }

        if (seg_frames.size() <= 1) {
            ++nsample;
            std::cout << std::endl;

            continue;
        }

        double inf = std::numeric_limits<double>::infinity();

        double min = inf;
        std::string argmin;

        double seg_loss = 0;

        std::vector<std::shared_ptr<autodiff::op_t>> outputs
            = reconstruct(comp_graph, seg_frames, var_tree, label, label_id, layer);

        for (int t = 0; t < outputs.size(); ++t) {
            la::tensor<double> gold { la::vector<double>(frames.at(t + 1)) };

            nn::l2_loss frame_loss {
                gold,
                autodiff::get_output<la::tensor_like<double>>(outputs.at(t))
            };

            seg_loss += frame_loss.loss();

            outputs.at(t)->grad = std::make_shared<la::tensor<double>>(
                frame_loss.grad());

        }

        std::cout << "loss: " << seg_loss << std::endl;

        auto topo_order = autodiff::natural_topo_order(comp_graph);
        autodiff::guarded_grad(topo_order, autodiff::grad_funcs);

        auto grad = rsg::make_tensor_tree(layer);

        tensor_tree::copy_grad(grad, var_tree);

        double n = tensor_tree::norm(grad);

        std::cout << "grad norm: " << n << std::endl;

        if (ebt::in(std::string("clip"), args)) {
            if (n > clip) {
                tensor_tree::imul(grad, clip / n);
                std::cout << "gradient clipped" << std::endl;
            }
        }

        auto vars = tensor_tree::leaves_pre_order(param);
        la::tensor<double> const& v = tensor_tree::get_tensor(vars[2]);

        double v1 = v.data()[0];

        opt->update(grad);

        double v2 = v.data()[0];

        std::cout << "weight: " << v1 << " update: " << v2 - v1
            << " rate: " << (v2 - v1) / v1 << std::endl;

        std::cout << "norm: " << tensor_tree::norm(param) << std::endl;

        std::cout << std::endl;

        ++nsample;
    }

    std::ofstream param_ofs { output_param };
    param_ofs << layer << std::endl;
    tensor_tree::save_tensor(param, param_ofs);
    param_ofs.close();

    std::ofstream opt_data_ofs { output_opt_data };
    opt_data_ofs << layer << std::endl;
    opt->save_opt_data(opt_data_ofs);
    opt_data_ofs.close();

}

std::vector<std::shared_ptr<autodiff::op_t>> reconstruct(
    autodiff::computation_graph& comp_graph,
    std::vector<std::shared_ptr<autodiff::op_t>> const& seg_frames,
    std::shared_ptr<tensor_tree::vertex> var_tree,
    std::string const& label,
    std::unordered_map<std::string, int> const& label_id,
    int layer)
{
    lstm::lstm_multistep_transcriber multistep;

    for (int i = 0; i < layer; ++i) {
        multistep.steps.push_back(std::make_shared<lstm::dyer_lstm_step_transcriber>(
            lstm::dyer_lstm_step_transcriber{}));
    }

    std::vector<std::shared_ptr<autodiff::op_t>> outputs;

    la::vector<double> label_vec;
    label_vec.resize(label_id.size());
    label_vec(label_id.at(label)) = 1;

    auto label_embed = autodiff::mul(comp_graph.var(la::tensor<double>(label_vec)),
        tensor_tree::get_var(var_tree->children[0]));

    std::shared_ptr<autodiff::op_t> frame = seg_frames.front();

    for (int i = 0; i < seg_frames.size(); ++i) {
        la::vector<double> dur_vec;
        dur_vec.resize(100);
        dur_vec(seg_frames.size() - i) = 1;

        auto dur_embed = autodiff::mul(comp_graph.var(la::tensor<double>(dur_vec)),
            tensor_tree::get_var(var_tree->children[1]));
        auto acoustic_embed = autodiff::mul(seg_frames.at(i),
            tensor_tree::get_var(var_tree->children[2]));

        auto input_embed = autodiff::add(
            std::vector<std::shared_ptr<autodiff::op_t>>{ label_embed,
                dur_embed, acoustic_embed,
                tensor_tree::get_var(var_tree->children[3]) });

        auto output = multistep(var_tree->children[4], input_embed);

        outputs.push_back(autodiff::add(autodiff::mul(output,
            tensor_tree::get_var(var_tree->children[5])),
            tensor_tree::get_var(var_tree->children[6])));

        frame = outputs.back();
    }

    return outputs;
}

std::string load_label_batch(std::ifstream& ifs)
{
    std::string area = "head";
    std::string line;
    std::string result;

    while (std::getline(ifs, line)) {
        if (area == "body" && line != ".") {
            result = line;
        }

        if (area == "head") {
            area = "body";
        }

        if (line == ".") {
            break;
        }
    }

    return result;
}

