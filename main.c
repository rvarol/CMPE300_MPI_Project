/* Student Number: Rahmetullah Varol
 * Student Number: 2016700159
 * Compile Status: Compiling
 * Program Status: Working
 * Notes: -
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Boolean type definition for ease of reading
typedef int bool;
#define true 1
#define false 0

/* STRUCTURE DEFINITIONS */
// Image structure
typedef struct Image {
	int** data;
	int cols;
	int rows;
} Image;

// Kernel structure
typedef struct Kernel {
	double** elements;
	int size;
} Kernel;

/*** HELPER METHODS ***/
// Read the image from text file to the given Image
void readImage(Image* inputImage, char* fileName);

// Print the given Image object to a text file
void writeImage(Image* writeImage, char* fileName);

// Print the given Image object to a text file (using the given tag)
void writeImageTagged(Image* writeImage, char* fileName, int tag);

// Build a Kernel structure from the given array
void buildKernel(Kernel* kernel, double* elems, int size);

// Apply smoothing kernel on the image
void applyBlur(Image* oldImage, Image* newImage);

// Apply the given Kernel to the given Image
void applyKernel(Image* oldImage, Image* newImage, Kernel* kernel);

// Take union of thresholding results of each given Image
void multiThreshold(Image* newImage, Image** thresholdImages, int thresholdImageCount, int threshold);

// Receive the data for the given Image with appropriate size from the given source 
void receiveImage(Image* inputImage, int source, int rowCount, int colCount);

// Append the given Image line to the back of the Image
void appendImageBack(Image *inputImage, int* line);

// Append the given Image line to the front of the Image
void appendImageFront(Image *inputImage, int* line);

// Initialize the Image with the given Image line
void appendEmptyImageBack(Image *inputImage, int* line, int lineSize);

// Free the memory allocated for the image
void freeImage(Image *inputImage);

// Free the memory allocated for the kernel
void freeKernel(Kernel *inputKernel);

int main(int argc, char** argv) {
	int rank = 0, size = 0;

	// Inititalize the MPI environment
	MPI_Init(&argc, &argv);

	// Check for incorrect input
	if (argc < 4) {
		printf("Usage: mpiexec -n <n_of_processors> <executable_name> <in_file> <out_file> <threshold>\n");	

		// End the MPI job
		MPI_Finalize();

		// Return with an error code
		return -1;
	}

	// Assign the input parameters
	char* inputFile = argv[1];
	char* outputFile = argv[2];
	int threshold = atoi(argv[3]);

	// Get the number of processors
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Get the rank of the current processor
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Common variables and structures
	Image blurImage, edgeImage; // Image structures
	int i = 0, j = 0, k = 0; // Iteration variables
	int message[2], partSize = 0, rowCount = 0, colCount = 0;
	bool outputSmooth = false; // If set, the smoothed image will be merged and written out

	/*** PART 1: READ AND DISTRIBUTE THE IMAGE ***/
	// Let the master process read and distribute the image
	if (rank == 0) {
		// The initial image will be read into this from the input file
		Image initialImage;

		// Read the image from the given text file
		readImage(&initialImage, inputFile);

		// Partition and send the image to the slave processes
		partSize = initialImage.cols/(size-1); // Row count of the partitioned images for each processor
		for (i = 1; i < size; i++) { // For each slave processor
			for (j = 0; j < partSize; j++) {
				// (SEND 1) MASTER -> SLAVE[i] : A single line of the partitioned image
				MPI_Send(&initialImage.data[((i-1)*partSize)+j][0], initialImage.rows, MPI_INT, i, 0, MPI_COMM_WORLD);
			}
		}

		// Determine row and column count of the initial image
		message[0] = initialImage.rows; // Row count
		message[1] = initialImage.cols; // Column count

		// Collect garbages
		//freeImage(&initialImage); // CAUTION!!
	}

	// Broadcast the row and column count of the initial image
	MPI_Bcast(message, 2, MPI_INT, 0, MPI_COMM_WORLD);

	rowCount = message[0]; // Calculate the row count of the initial image
	colCount = message[1]; // Calculate the column count of the initial image
	partSize = rowCount/(size-1); // Calculate the partition size in slave processors

	if (rank != 0) {
		/*** PART 2: RECEIVE EXTEND THE PARTIAL IMAGES ***/
		// The partial image sent from the master will be read into this
		Image partImage;

		// (RECV 1) MASTER -> SLAVE(rank) : A single line of the partitioned image
		// Receive the partial image from the master one line at a time
		receiveImage(&partImage, 0, partSize, colCount);

		// (SEND 2) SLAVE(rank) -> SLAVE(rank-1), SLAVE(rank+1) : Rows at the common borders
		// Send the border rows to adjacent processes
		if (rank == 1) { 
			// For the first processor send only the last row
			MPI_Send(&partImage.data[partSize-1][0], colCount, MPI_INT, 2, 0, MPI_COMM_WORLD);
		} else if (rank == size-1) { 
			// For the last processor send only the first row
			MPI_Send(&partImage.data[0][0], colCount, MPI_INT, rank-1, 0, MPI_COMM_WORLD);
		} else { 
			// For all other processor send both the first and last rows
			MPI_Send(&partImage.data[0][0], colCount, MPI_INT, rank-1, 0, MPI_COMM_WORLD);
			MPI_Send(&partImage.data[partSize-1][0], colCount, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
		}
		
		// (RECV 2) SLAVE(rank) -> SLAVE(rank-1), SLAVE(rank+1) : Rows at the common borders
		// Receive the border rows from adjacent processes and append them to the image accordingly
		int* readBuffer = malloc(colCount*sizeof(int));
		if (rank == 1) {
			MPI_Recv(readBuffer, colCount, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			appendImageBack(&partImage, readBuffer);
		} else if (rank == size-1) {
			MPI_Recv(readBuffer, colCount, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			appendImageFront(&partImage, readBuffer);
		} else {
			MPI_Recv(readBuffer, colCount, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			appendImageFront(&partImage, readBuffer);

			MPI_Recv(readBuffer, colCount, MPI_INT, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			appendImageBack(&partImage, readBuffer);
		}
		free(readBuffer);

		/*** PART 3: SMOOTH THE EXTENDED PARTIAL IMAGES ***/
		// Create the kernel to smooth the images
		applyBlur(&partImage, &blurImage);

		// Collect garbages
		freeImage(&partImage);
	}

	// OPTIONAL: If desired the smoothed image is written out
	if (outputSmooth) {
		if (rank != 0) {
			// Send all the rows of the smoothed image to the master
			for (i = 0; i < blurImage.rows; i++) {
				MPI_Send(&blurImage.data[i][0], blurImage.cols, MPI_INT, 0, 0, MPI_COMM_WORLD);
			}
		}

		if (rank == 0) {
			// Partially smoothed images will be merged into this
			Image finalBlurImage;

			int* readBuffer = malloc((colCount-2)*sizeof(int));
			MPI_Recv(readBuffer, colCount-2, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			appendEmptyImageBack(&finalBlurImage, readBuffer, colCount-2);
			
			// Receive the partial smoothed images from the slaves with the proper order
			for (i = 0; i < partSize-2; i++) {
				MPI_Recv(readBuffer, colCount-2, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				appendImageBack(&finalBlurImage, readBuffer);
			}

			for (k = 2; k < size-1; k++) {
				for (i = 0; i < partSize; i++) {
					MPI_Recv(readBuffer, colCount-2, MPI_INT, k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					appendImageBack(&finalBlurImage, readBuffer);
				}	
			}

			for (i = 0; i < partSize-1; i++) {
				MPI_Recv(readBuffer, colCount-2, MPI_INT, size-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				appendImageBack(&finalBlurImage, readBuffer);
			}

			writeImage(&finalBlurImage, "finalSmooth.txt");

			// Collect garbages
			free(readBuffer);

			//freeImage(&finalBlurImage); // CAUTION!!!
		}
	}

	// Synchronize the processes
	MPI_Barrier(MPI_COMM_WORLD);

	/*** PART 4: EXTEND THE SMOOTHED PARTIAL IMAGES ***/
	if (rank != 0) {
		// (SEND 3) SLAVE(rank) -> SLAVE(rank-1), SLAVE(rank+1) : Rows of the smoothed image at the common borders
		// Send the border rows of the smoothed images to adjacent processes
		if (rank == 1) {
			// For the first processor send only the last row
			MPI_Send(&blurImage.data[blurImage.rows-1][0], blurImage.cols, MPI_INT, 2, 0, MPI_COMM_WORLD); // Send the last row
		} else if (rank == size-1) {
			// For the last processor send only the first row
			MPI_Send(&blurImage.data[0][0], blurImage.cols, MPI_INT, rank-1, 0, MPI_COMM_WORLD); // Send the first row
		} else {
			// For all other processor send both the first and last rows
			MPI_Send(&blurImage.data[0][0], blurImage.cols, MPI_INT, rank-1, 0, MPI_COMM_WORLD); // Send the first row
			MPI_Send(&blurImage.data[blurImage.rows-1][0], blurImage.cols, MPI_INT, rank+1, 0, MPI_COMM_WORLD); // Send the last row
		}

		// (RECV 3) SLAVE(rank) -> SLAVE(rank-1), SLAVE(rank+1) : Rows at the common borders
		// Receive the border rows from adjacent processes and append them to the image accordingly
		int* readBuffer = malloc((blurImage.cols)*sizeof(int));
		if (rank == 1) {
			MPI_Recv(readBuffer, blurImage.cols, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			appendImageBack(&blurImage, readBuffer);
		} else if (rank == size-1) {
			MPI_Recv(readBuffer, blurImage.cols, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			appendImageFront(&blurImage, readBuffer);
		} else {
			MPI_Recv(readBuffer, blurImage.cols, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			appendImageFront(&blurImage, readBuffer);

			MPI_Recv(readBuffer, blurImage.cols, MPI_INT, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			appendImageBack(&blurImage, readBuffer);
		}
		free(readBuffer);
	}

	/*** PART 5: APPLY EDGE FILTER TO THE EXTENDED IMAGES ***/
	if (rank != 0 && !((rank == 1 || rank == size-1) && size == 101)) {
		// These will be used for applying the edge detection kernels
		Image lineImage1, lineImage2, lineImage3, lineImage4;
		Kernel lineKernel1, lineKernel2, lineKernel3, lineKernel4;

		// Create the edge detection kernels
		double lineElements1[9] = {-1, -1, -1, 2, 2, 2, -1, -1, -1};
		double lineElements2[9] = {-1, 2, -1, -1, 2, -1, -1, 2, -1};
		double lineElements3[9] = {-1, -1, 2, -1, 2, -1, 2, -1, -1};
		double lineElements4[9] = {2, -1, -1, -1, 2, -1, -1, -1 ,2};

		buildKernel(&lineKernel1, lineElements1, 3);
		buildKernel(&lineKernel2, lineElements2, 3);
		buildKernel(&lineKernel3, lineElements3, 3);
		buildKernel(&lineKernel4, lineElements4, 3);

		// Apply the edge detection kernels
		applyKernel(&blurImage, &lineImage1, &lineKernel1);
		applyKernel(&blurImage, &lineImage2, &lineKernel2);
		applyKernel(&blurImage, &lineImage3, &lineKernel3);
		applyKernel(&blurImage, &lineImage4, &lineKernel4);

		// Apply threshold to the line images
		Image* thresholdImages[4] = { &lineImage1, & lineImage2, &lineImage3, &lineImage4 };
		multiThreshold(&edgeImage, thresholdImages, 4, threshold);

		// Collect garbages
		freeImage(&blurImage);
		freeImage(&lineImage1);
		freeImage(&lineImage2);
		freeImage(&lineImage3);
		freeImage(&lineImage4);
		freeKernel(&lineKernel1);
		freeKernel(&lineKernel2);
		freeKernel(&lineKernel3);
		freeKernel(&lineKernel4);
	}

	// Synchronize the processes
	MPI_Barrier(MPI_COMM_WORLD);

	/*** PART 6: MERGE THE EDGE IMAGES ***/
	if (rank != 0 && !((rank == 1 || rank == size-1) && size == 101)) {
		// (SEND 4) SLAVE(rank) -> MASTER : All the rows of the partial edge image
		// Send all the rows of the partial edge image to the master
		for (i = 0; i < edgeImage.rows; i++) {
			MPI_Send(&edgeImage.data[i][0], edgeImage.cols, MPI_INT, 0, 0, MPI_COMM_WORLD);
		}

		// Collect garbages
		freeImage(&edgeImage);
	}

	if (rank == 0) {
		// The partial edge images sent from the slaves will be merged into this
		Image finalImage;
		int firstSlave = 1;

		if (size == 101) {
			firstSlave = 2;
		}

		// (RECV 4) SLAVE(rank) -> MASTER : All the rows of the partial edge image
		// Receive all the rows of the partial edge images from the slave processes
		int* readBuffer = malloc((colCount-4)*sizeof(int));
		MPI_Recv(readBuffer, colCount-4, MPI_INT, firstSlave, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		appendEmptyImageBack(&finalImage, readBuffer, colCount-4);
		
		// Receive the partial smoothed images from the slaves with the proper order add them into the final image
		if (!((rank == 1 || rank == size-1) && size == 101)) {
			for (i = 0; i < partSize-3; i++) {
				MPI_Recv(readBuffer, colCount-4, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				appendImageBack(&finalImage, readBuffer);
			}
		}

		for (k = firstSlave+1; k < size-1; k++) {
			for (i = 0; i < partSize; i++) {
				MPI_Recv(readBuffer, colCount-4, MPI_INT, k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				appendImageBack(&finalImage, readBuffer);
			}	
		}

		if (!((rank == 1 || rank == size-1) && size == 101)) {
			for (i = 0; i < partSize-2; i++) {
				MPI_Recv(readBuffer, colCount-4, MPI_INT, size-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				appendImageBack(&finalImage, readBuffer);
			}
		}

		// Output the final image
		writeImage(&finalImage, outputFile);

		// Collect garbages
		free(readBuffer);
		freeImage(&finalImage);
	}


	// End the MPI job
	MPI_Finalize();

	return 0;
}

// Read the image from text file to the given Image
void readImage(Image* inputImage, char* fileName) {
	FILE* fp = fopen(fileName, "r");

	int colCount = 0, rowCount = 0, i = 0, j = 0;
	char num[4], c;
	bool countSpaces = true;

	// Determine the row and column size of the image
	while ((c = fgetc(fp)) != EOF) {
		if (c == ' ' && countSpaces) {
			colCount++;
		} else if (c == '\n') {
			rowCount++;
			countSpaces = false;
		}
	}
	rewind(fp);

	// Initialize the image structure
	inputImage->rows = rowCount;
	inputImage->cols = colCount;
	inputImage->data = (int**)malloc(rowCount*sizeof(int*));

	// Allocate space in each row for the columns of the image
	for (i = 0; i < rowCount; i++) {
		inputImage->data[i] = (int*)malloc(colCount*sizeof(int));
	}

	// Read the rows and columns into the image structure
	int row = 0, col = 0;
	while ((c = fgetc(fp)) != EOF) {
		if (c == ' ') {
			inputImage->data[row][col++] = atoi(num);

			j = 0;
			num[0] = '\0'; num[1] = '\0'; num[2] = '\0'; num[3] = '\0';
		} else if (c == '\n') {
			col = 0;
			row++;
			j = 0;
		}

		num[j++] = c;
	}
}

// Print the given Image object to a text file
void writeImage(Image* writeImage, char* fileName) {
	int i = 0, j = 0;
	FILE* fp = fopen(fileName, "w");

	// Write the image data into the text file
	for (i = 0; i < writeImage->rows; i++) {
		for (j = 0; j < writeImage->cols; j++) {
			fprintf(fp, "%d ", writeImage->data[i][j]);
		}
		fprintf(fp, "\n");
	}
}

// Print the given Image object to a text file (using the given tag)
void writeImageTagged(Image* inputImage, char* fileName, int tag) {
	// Construct the output file name and call the image write procedure
	const int MAX_FILENAME = 100;
	char outFilename[MAX_FILENAME];
	sprintf(outFilename, "%s%d.txt", fileName, tag);
	writeImage(inputImage, outFilename);
}

// Build a Kernel structure from the given array
void buildKernel(Kernel* kernel, double* elems, int size) {
	int i = 0, j = 0;
	kernel->size = size;
	
	kernel->elements = malloc(size*sizeof(double*));

	// Fill the kernel data using the given double array
	for (i = 0; i < size; i++) {
		kernel->elements[i] = malloc(size*sizeof(double));
		for (j = 0; j < size; j++) {
			kernel->elements[i][j] = elems[i*size+j];
		}
	}
}

// Apply smoothing kernel on the image
void applyBlur(Image* oldImage, Image* newImage) {
	int i = 0, j = 0, k1 = 0, k2 = 0;

	// Initialize the new image structure
	newImage->cols = oldImage->cols-2;
	newImage->rows = oldImage->rows-2;
	newImage->data = malloc((newImage->rows)*sizeof(int*));

	// Iterate over the old image and for each pixel calcualte the average of its neighborhood
	for (i = 1; i < oldImage->rows-1; i++) {
		newImage->data[i-1] = malloc(newImage->cols*sizeof(int));
		for (j = 1; j < oldImage->cols-1; j++) {
			double total = 0;
			for (k1 = 0; k1 < 3; k1++) {
				for (k2 = 0; k2 < 3; k2++) {
					total += oldImage->data[(i-1)+k1][(j-1)+k2];
				}
			}
			newImage->data[i-1][j-1] = total/(9.0);
		}
	}
}

// Apply the given Kernel to the given Image
void applyKernel(Image* oldImage, Image* newImage, Kernel* kernel) {
	int i = 0, j = 0, k1 = 0, k2 = 0;

	// Initialize the new image structure
	newImage->cols = oldImage->cols-2;
	newImage->rows = oldImage->rows-2;
	newImage->data = malloc((newImage->rows)*sizeof(int*));
	
	// Iterate over the old image and for each pixel apply convolution using the given Kernel
	for (i = 1; i < oldImage->rows-1; i++) {
		newImage->data[i-1] = malloc(newImage->cols*sizeof(int));
		for (j = 1; j < oldImage->cols-1; j++) {
			double value = 0.0;
			for (k1 = 0; k1 < 3; k1++) {
				for (k2 = 0; k2 < 3; k2++) {
					double imageValue = oldImage->data[(i-1)+k1][(j-1)+k2];
					value += kernel->elements[k1][k2]*imageValue;
				}
			}
			newImage->data[i-1][j-1] = (int)(value);
		}
	}
}

// Take union of thresholding results of each given Image
void multiThreshold(Image* newImage, Image** thresholdImages, int thresholdImageCount, int threshold) {
	int i = 0, j = 0, k = 0;

	// Initialize the new image structure
	newImage->rows = thresholdImages[0]->rows;
	newImage->cols = thresholdImages[0]->cols;
	newImage->data = malloc((newImage->rows)*sizeof(int*));

	// Iterate over the given threshold images and OR their thresholding results
	for (i = 0; i < newImage->rows; i++) {
		newImage->data[i] = malloc((newImage->cols)*sizeof(int*));
		for (j = 0; j < newImage->cols; j++) {
			newImage->data[i][j] = 0;
			for (k = 0; k < thresholdImageCount; k++) {
				if (thresholdImages[k]->data[i][j] > threshold) {
					newImage->data[i][j] = 255;
				}
			}
		}
	}
}

// Receive the data for the given Image with appropriate size from the given source
void receiveImage(Image* inputImage, int source, int rowCount, int colCount) {
	int i = 0;

	// Initialize the image structure
	inputImage->rows = rowCount;
	inputImage->cols = colCount;
	inputImage->data = malloc(rowCount*sizeof(int*));

	// Receive the lines from the source one line at a time
	for (i = 0; i < rowCount; i++) {
		inputImage->data[i] = malloc((colCount)*sizeof(int));
		MPI_Recv(inputImage->data[i], colCount, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}	
}

// Initialize the Image with the given Image line
void appendEmptyImageBack(Image *inputImage, int* line, int lineSize) {
	// Initialize the image structure and append the line to the new image
	inputImage->rows = 0;
	inputImage->cols = lineSize;
	appendImageBack(inputImage, line);
}

// Append the given Image line to the back of the Image
void appendImageBack(Image *inputImage, int* line) {
	int i = 0;

	// Reallocate memory for the image structure (add one more row)
	inputImage->data = realloc(inputImage->data, (inputImage->rows+1)*sizeof(int*));
	inputImage->data[inputImage->rows] = malloc((inputImage->cols)*sizeof(int*));

	// Populate the new row with the given integer array
	for (i = 0; i < inputImage->cols; i++) {
		inputImage->data[inputImage->rows][i] = line[i];
	}

	// Increase the row count
	inputImage->rows++;
}

// Append the given Image line to the front of the Image
void appendImageFront(Image *inputImage, int* line) {
	int i = 0;

	// Reallocate memory for the image structure (add one more row and shift old rows)
	inputImage->data = realloc(inputImage->data, (inputImage->rows+1)*sizeof(int*));
	for (i = inputImage->rows; i > 0; i--) {
		inputImage->data[i] = inputImage->data[i-1];
	}
	inputImage->data[0] = malloc((inputImage->cols)*sizeof(int*));
	inputImage->rows++;

	// Populate the new row with the given integer array
	for (i = 0; i < inputImage->cols; i++) {
		inputImage->data[0][i] = line[i];
	}
}

// Free the memory allocated for the image
void freeImage(Image *inputImage) {
	int i = 0;

	// Free the memory allocated for each row and finally for the row pointers
	for (i = 0; i < inputImage->rows; i++) {
		free(inputImage->data[i]);
	}
	free(inputImage->data);

	inputImage->rows = 0;
	inputImage->cols = 0;
}

// Free the memory allocated for the kernel
void freeKernel(Kernel *inputKernel) {
	int i = 0;

	// Free the memory allocated for each row and finally for the row pointers
	for (i = 0; i < inputKernel->size; i++) {
		free(inputKernel->elements[i]);
	}
	free(inputKernel->elements);
}