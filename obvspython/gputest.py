import torch as pt

def main():

    # Print CUDA details
    print("CUDA available:" + str(pt.cuda.is_available()))
    print("CUDA device count:" + str(pt.cuda.device_count()))

    # Create random input and output tensors
    x = pt.randn(3, 3)
    y = pt.randn(3, 3)

    if pt.cuda.is_available():
        # Move tensors to GPU
        x = x.cuda()
        y = y.cuda()

        # Perform matrix multiplication on GPU
        z = x @ y
        print("Matrix multiplication result on GPU:\n")
        print(z)

        # Move result back to CPU
        z = z.cpu()
    else:
        # Perform computation on CPU
        z = x @ y
        print("Matrix multiplication result on CPU:\n")
        print(z)

if __name__ == "__main__":
    main()