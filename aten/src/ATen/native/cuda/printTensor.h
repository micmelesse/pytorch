void print_tensor(const Tensor& in) {
  bool is_complex = in.is_complex();
  void* in_data_ptr = in.data_ptr();
  int length = in.numel();

  HIP_vector_type<float, 2>* in_cast_complex;
  float* in_cast_real;

  if (is_complex) {
    in_cast_complex = static_cast<HIP_vector_type<float, 2>*>(in_data_ptr);
  } else {
    in_cast_real = static_cast<float*>(in_data_ptr);
  }

  for (size_t i = 0; i < length; i++) {
    if (is_complex) {
      printf("%f,%f ", in_cast_complex[i].x, in_cast_complex[i].y);
    } else {
      printf("%f ", in_cast_real[i]);
    }
  }

  printf("\n");
  in.print();
}

void print_buffer(void* in, int length, bool is_complex) {
  void* in_data_ptr = in;

  HIP_vector_type<float, 2>* in_cast_complex;
  float* in_cast_real;

  if (is_complex) {
    in_cast_complex = static_cast<HIP_vector_type<float, 2>*>(in_data_ptr);
  } else {
    in_cast_real = static_cast<float*>(in_data_ptr);
  }

  for (size_t i = 0; i < length; i++) {
    if (is_complex) {
      printf("%f,%f ", in_cast_complex[i].x, in_cast_complex[i].y);
    } else {
      printf("%f ", in_cast_real[i]);
    }
  }

  printf("\n");
}
