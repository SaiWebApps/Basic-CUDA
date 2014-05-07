ARCH20="-arch=sm_20"

all:
	make normal_reduction
	make tree_reduction	

normal_reduction:
	nvcc ${ARCH20} reduction.cu -o normal.out

tree_reduction:
	nvcc tree_reduction.cu -o tree.out

clean:
	rm *.out
