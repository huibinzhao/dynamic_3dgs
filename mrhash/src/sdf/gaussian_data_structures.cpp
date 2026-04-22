#include "gaussian_data_structures.cuh"

namespace cupanutils {
  namespace cugeoutils {
    template <typename T>
    GaussianContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::GaussianContainer(
      std::string gs_optimization_param_path) :
      gs_model_(gs::param::read_optim_params_from_json(gs_optimization_param_path), ".") {
      CUDA_CHECK(cudaMalloc((void**) &d_qtree_nodes_, sizeof(gs::CUDANode) * 1000000));
      CUDA_CHECK(cudaMalloc((void**) &d_num_qtree_nodes_, sizeof(size_t)));
      CUDA_CHECK(cudaMalloc((void**) &d_num_valid_qtree_nodes_, sizeof(uint)));
      CUDA_CHECK(cudaMalloc((void**) &d_positions_, sizeof(gs::CUDANode) * 1000000));
      CUDA_CHECK(cudaMalloc((void**) &d_colors_, sizeof(gs::CUDANode) * 1000000));
      CUDA_CHECK(cudaMalloc((void**) &d_scales_, sizeof(gs::CUDANode) * 1000000));
    }

    template <typename T>
    GaussianContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::~GaussianContainer() {
      CUDA_CHECK(cudaFree(d_qtree_nodes_));
      CUDA_CHECK(cudaFree(d_num_qtree_nodes_));
      CUDA_CHECK(cudaFree(d_num_valid_qtree_nodes_));
      CUDA_CHECK(cudaFree(d_positions_));
      CUDA_CHECK(cudaFree(d_colors_));
      CUDA_CHECK(cudaFree(d_scales_));
    }

    template <typename T>
    void GaussianContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::setupGSCamera(Camera& camera) {
      /* SETUP GS Camera */
      Eigen::Matrix4f T_SW     = Eigen::Isometry3f(CUDA2Eig(camera.camInWorld())).inverse().matrix();
      torch::Tensor W2C_matrix = torch::from_blob(T_SW.data(), {4, 4}, torch::kFloat).clone().to(torch::kCUDA, true);
      torch::Tensor proj_matrix =
        gs::getProjectionMatrix(camera.cols(), camera.rows(), camera.fx(), camera.fy(), camera.cx(), camera.cy())
          .to(torch::kCUDA, true);

      cur_gs_cam_.height           = camera.rows();
      cur_gs_cam_.width            = camera.cols();
      cur_gs_cam_.T_W2C            = W2C_matrix;
      cur_gs_cam_.fov_x            = camera.hfov();
      cur_gs_cam_.fov_y            = camera.vfov();
      cur_gs_cam_.T_W2C            = W2C_matrix;
      cur_gs_cam_.full_proj_matrix = W2C_matrix.mm(proj_matrix);
      cur_gs_cam_.cam_center       = W2C_matrix.inverse()[3].slice(0, 0, 3);
      gs_cam_list_.push_back(cur_gs_cam_);
    }

    template <typename T>
    void GaussianContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::extractNodesQTree(
      Camera& camera,
      VoxelContainer<T>& container,
      cupanutils::cugeoutils::CUDAMatrixuc3& rgb_img,
      cupanutils::cugeoutils::CUDAMatrixf& depth_img,
      cupanutils::cugeoutils::CUDAMatrixb* dynamic_mask) {
  gs::CUDAQTree qtree(gs_model_.optimParams.qtree_thresh,
          gs_model_.optimParams.qtree_min_pixel_size,
          d_qtree_nodes_,
          rgb_img);
      qtree.subdivide();
      num_qtree_nodes_ = qtree.getNumLeaves();
      CUDA_CHECK(cudaMemcpy(d_num_qtree_nodes_, &num_qtree_nodes_, sizeof(size_t), cudaMemcpyHostToDevice));

      torch::Tensor image_tensor = torch::from_blob(reinterpret_cast<uint8_t*>(rgb_img.data<1>()),
                                                    {rgb_img.rows(), rgb_img.cols(), 3},
                                                    {rgb_img.cols() * 3, 3, 1},
                                                    torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
      cur_gt_img_                = image_tensor.to(torch::kFloat32).permute({2, 0, 1}).clone() / 255.f;
      cur_gt_img_                = torch::clamp(cur_gt_img_, 0.f, 1.f);
      gt_img_list_.push_back(cur_gt_img_);

      // Build mask tensor for training: 1 = static (use), 0 = dynamic (ignore)
      if (dynamic_mask != nullptr && dynamic_mask->size() > 0) {
        dynamic_mask->toHost();
        auto mask_tensor = torch::ones({1, dynamic_mask->rows(), dynamic_mask->cols()},
                                       torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        auto mask_accessor = mask_tensor.accessor<float, 3>();
        for (int r = 0; r < dynamic_mask->rows(); ++r) {
          for (int c = 0; c < dynamic_mask->cols(); ++c) {
            if (dynamic_mask->at(r, c)) {
              mask_accessor[0][r][c] = 0.f;
            }
          }
        }
        cur_mask_tensor_ = mask_tensor.to(torch::kCUDA);
      } else {
        cur_mask_tensor_ = torch::ones({1, rgb_img.rows(), rgb_img.cols()},
                                       torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
      }
      mask_list_.push_back(cur_mask_tensor_);
    }

    template <typename T>
    void GaussianContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::optimizeGS() {
      bool isKeyframe = false;

      // Update keyframe list
      if (num_valid_qtree_nodes_ > gs_model_.optimParams.kf_thresh) {
        isKeyframe = true;
      } else {
        // Only keep non-keyframes for ScanNet++ dataset
        if (!gs_model_.optimParams.keep_all_frames) {
          gs_cam_list_.pop_back();
          gt_img_list_.pop_back();
          mask_list_.pop_back();
        }
      }

      if (num_valid_qtree_nodes_ != 0) {
        torch::NoGradGuard no_grad;
        gs_model_.Add_gaussians(d_positions_, d_colors_, d_scales_, num_valid_qtree_nodes_);
      }

      int iters = gs_model_.optimParams.kf_iters;
      if (!isKeyframe) {
        iters = gs_model_.optimParams.non_kf_iters;
      }

      std::vector<int> kf_indices = gs::get_random_indices(gt_img_list_.size());

      // Start online optimization
      for (int iter = 0; iter < iters; iter++) {
        auto [image, viewspace_point_tensor, visibility_filter, radii] = gs::render(cur_gs_cam_, gs_model_);

        // Loss Computations: mask out dynamic regions
        auto masked_image = image * cur_mask_tensor_;
        auto masked_gt    = cur_gt_img_ * cur_mask_tensor_;
        auto loss         = gs::l1_loss(masked_image, masked_gt);

        // Optimization
        loss.backward();
        gs_model_.optimizer->step();
        gs_model_.optimizer->zero_grad(true);

        // Store the cv::Mat rendered image for visualization
        if (iter == iters - 1) {
          auto rendered_img_tensor = image.detach().permute({1, 2, 0}).contiguous().to(torch::kCPU);
          rendered_img_tensor      = rendered_img_tensor.mul(255).clamp(0, 255).to(torch::kU8);
          cv::Mat temp(image.size(1), image.size(2), CV_8UC3, rendered_img_tensor.data_ptr());
          last_rendered_img_ = temp.clone();
          has_rendered_img_ = true;
        }
      }

      if (!isKeyframe) {
        int kf_iters = gs_model_.optimParams.random_kf_num;
        if (kf_indices.size() < kf_iters) {
          kf_iters = kf_indices.size();
        }
        for (int i = 0; i < kf_iters; i++) {
          auto kf_gt_img  = gt_img_list_[kf_indices[i]];
          auto kf_gs_cam  = gs_cam_list_[kf_indices[i]];
          auto kf_mask    = mask_list_[kf_indices[i]];

          auto [image, viewspace_point_tensor, visibility_filter, radii] = gs::render(kf_gs_cam, gs_model_);
          auto masked_image = image * kf_mask;
          auto masked_gt    = kf_gt_img * kf_mask;
          auto loss         = gs::l1_loss(masked_image, masked_gt);
          loss.backward();
          gs_model_.optimizer->step();
          gs_model_.optimizer->zero_grad(true);
        }
      }
      torch::cuda::synchronize();
    }

    template <typename T>
    void
    GaussianContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::runGS(Camera& camera,
                                                                              VoxelContainer<T>& container,
                                                                              cupanutils::cugeoutils::CUDAMatrixuc3& rgb_img,
                                                                              cupanutils::cugeoutils::CUDAMatrixf& depth_img,
                                                                              cupanutils::cugeoutils::CUDAMatrixb* dynamic_mask) {
      size_t free_byte;
      size_t total_byte;
      cudaMemGetInfo(&free_byte, &total_byte);
      if (free_byte < 100 * 1024 * 1024) {
        std::cout << "[GaussianContainer] Low GPU memory (" << free_byte / (1024 * 1024)
                  << " MB free). Skipping Gaussian Splatting update to avoid OOM." << std::endl;
        return;
      }
      setupGSCamera(camera);
      extractNodesQTree(camera, container, rgb_img, depth_img, dynamic_mask);
      checkNodes(camera, container, rgb_img, depth_img, dynamic_mask);
      optimizeGS();
    }

    template <typename T>
    void GaussianContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::optimizeGSFinal() {
      auto lambda = gs_model_.optimParams.lambda_dssim;
      auto iters  = gs_model_.optimParams.global_iters;

      for (int it = 0; it < iters; it++) {
        std::vector<int> indices = gs::get_random_indices(gt_img_list_.size());
        for (int i = 0; i < indices.size(); i++) {
          auto cur_gt_img = gt_img_list_[indices[i]];
          auto cur_gs_cam = gs_cam_list_[indices[i]];
          auto cur_mask   = mask_list_[indices[i]];

          auto [image, viewspace_point_tensor, visibility_filter, radii] = gs::render(cur_gs_cam, gs_model_);

          // Loss Computations: mask out dynamic regions
          auto masked_image = image * cur_mask;
          auto masked_gt    = cur_gt_img * cur_mask;
          auto l1_loss   = gs::l1_loss(masked_image, masked_gt);
          auto ssim_loss = gs::ssim(masked_image, masked_gt, gs::conv_window, gs::window_size, gs::channel);
          auto loss      = (1.f - lambda) * l1_loss + lambda * (1.f - ssim_loss);

          // Optimization
          loss.backward();
          gs_model_.optimizer->step();
          gs_model_.optimizer->zero_grad(true);
        }
      }
      torch::cuda::synchronize();
    }

    template <typename T>
    void GaussianContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::renderOnly(Camera& camera) {
      // Set up the GS camera from the current pose (without adding to keyframe list)
      Eigen::Matrix4f T_SW     = Eigen::Isometry3f(CUDA2Eig(camera.camInWorld())).inverse().matrix();
      torch::Tensor W2C_matrix = torch::from_blob(T_SW.data(), {4, 4}, torch::kFloat).clone().to(torch::kCUDA, true);
      torch::Tensor proj_matrix =
        gs::getProjectionMatrix(camera.cols(), camera.rows(), camera.fx(), camera.fy(), camera.cx(), camera.cy())
          .to(torch::kCUDA, true);

      gs::Camera render_cam;
      render_cam.height           = camera.rows();
      render_cam.width            = camera.cols();
      render_cam.T_W2C            = W2C_matrix;
      render_cam.fov_x            = camera.hfov();
      render_cam.fov_y            = camera.vfov();
      render_cam.full_proj_matrix = W2C_matrix.mm(proj_matrix);
      render_cam.cam_center       = W2C_matrix.inverse()[3].slice(0, 0, 3);

      // Render without gradient computation
      torch::NoGradGuard no_grad;
      auto [image, viewspace_point_tensor, visibility_filter, radii] = gs::render(render_cam, gs_model_);

      // Store the rendered image for visualization
      auto rendered_img_tensor = image.detach().permute({1, 2, 0}).contiguous().to(torch::kCPU);
      rendered_img_tensor      = rendered_img_tensor.mul(255).clamp(0, 255).to(torch::kU8);
      cv::Mat temp(image.size(1), image.size(2), CV_8UC3, rendered_img_tensor.data_ptr());
      last_rendered_img_ = temp.clone();
      has_rendered_img_ = true;
      torch::cuda::synchronize();
    }

  } // namespace cugeoutils
} // namespace cupanutils
