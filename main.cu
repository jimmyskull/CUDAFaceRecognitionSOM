/* Copyright 2013 Paulo Roberto Urio <paulourio@gmail.com> */
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <dirent.h>
#include <unistd.h>

#include <algorithm>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

#define TRAINING_DIR "./training/"
#define TESTING_DIR "./testing/"

#define PEOPLE 10

struct Pattern {
	std::vector<cv::Point2f> winners;  // Training winners
};

#include "global.h"
#include "cutil.h"

extern __host__ cv::Point2f find_best(const uchar* input);
extern __host__ void initialize_map();
extern __host__ void update_neighborhood(const uchar* input, float epoch,
		cv::Point2f best);

float* devSOM;

static float epoch = 0.0;

static std::vector<std::string> training_files;
static std::vector<std::string> testing_files;

static std::vector<Pattern> patterns;

static void read_directory(const char *directory_path,
		std::vector<std::string>* output) {
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(directory_path)) != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			// printf("%s%s\n", directory_path,  ent->d_name);
			output->push_back(std::string(ent->d_name));
		}
		closedir(dir);
	} else {
		perror("Could not open directory.");
		exit(EXIT_FAILURE);
	}
}

static cv::gpu::GpuMat LoadImage(const char* path, const std::string& name) {
	std::stringstream file;
	file << path << name;
	std::string filename = file.str();
	if (filename.substr(filename.find_last_of(".") + 1) != "pgm")
		return cv::gpu::GpuMat();
	if (access(filename.c_str(), F_OK) == -1)
		return cv::gpu::GpuMat();
	//fprintf(stderr, "%s\n", filename.c_str());
	cv::Mat input = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (input.empty()) {
		printf("Erro ao carregar imagem!\n");
		exit(-1);
	}
	cv::gpu::GpuMat dev_input(input);
	cv::gpu::GpuMat dev_image;

	cv::gpu::blur(dev_input, dev_image, cv::Size(3, 3));
	cv::gpu::Canny(dev_image, dev_image, 35, 100, 3);
	return dev_image;
}

static void Train() {
	// Init SOM
	fprintf(stderr, "Initializing...\n");
	initialize_map();
	// Train
	fprintf(stderr, "Training...\n");
	epoch = 0.0;
	size_t T = training_files.size();
	while (epoch < ITERATION_NUMBER) {
		fprintf(stderr, "\r%d/%d", (int) epoch + 1, (int) ITERATION_NUMBER);
		std::random_shuffle(training_files.begin(), training_files.end());
		for (size_t i = 0; i < T; ++i) {
			cv::gpu::GpuMat dev_image(
					LoadImage(TRAINING_DIR, training_files[i]));
			if (dev_image.empty())
				continue;
			cv::Point2f best = find_best(dev_image.data);
			update_neighborhood(dev_image.data, epoch, best);
		}
		epoch += 1.0;
	}
	fprintf(stderr, "\nComputing patterns...\n");
	for (size_t i = 0; i < T; ++i) {
		const std::string& name = training_files[i];
		cv::gpu::GpuMat dev_image(LoadImage(TRAINING_DIR, name));
		if (dev_image.empty())
			continue;
		cv::Point2f best = find_best(dev_image.data);

		int person;
		sscanf(name.c_str(), "%d", &person);
		patterns[person].winners.push_back(best);
	}
	fprintf(stderr, "Finished.\n");
}

void Test() {
	fprintf(stderr, "Testing...\n");
	size_t T = testing_files.size();
	size_t correct_count = 0;
	size_t total = 0;
	for (size_t i = 0; i < T; ++i) {
		const std::string& name = testing_files[i];
		cv::gpu::GpuMat dev_image(LoadImage(TESTING_DIR, name));
		if (dev_image.empty())
			continue;
		cv::Point2f best = find_best(dev_image.data);

		int person;
		sscanf(name.c_str(), "%d", &person);

		int winner = 0;
		float better = FLT_MAX;

		++total;
		bool stop = false;
		for (int p = 0; p < PEOPLE && !stop; ++p) {
			Pattern& pattern = patterns[p];
			for (int j = 0; j < pattern.winners.size() && !stop; ++j) {
				cv::Point2f w = pattern.winners[j];
				if ((int) w.x == (int) best.x && (int) w.y == (int) best.y) {
					winner = person;
					stop = true;
				} else {
					float x = w.x - best.x;
					float y = w.y - best.y;
					float distance = std::sqrt(x * x + y * y);
					if (distance < better) {
						winner = person;
						better = distance;
					}
				}
			}
		}
		if (winner == person) {
			++correct_count;
		}
	}
	printf("Total: %zd/%zd\n", correct_count, total);
}

int main(void) {
	cudaDeviceReset();
	cv::gpu::setDevice(0);
	cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

	patterns.resize(PEOPLE);
	cudaSafeCall(cudaMalloc(&devSOM, N * N * DIM * DIM * sizeof(float)));

	read_directory(TRAINING_DIR, &training_files);
	read_directory(TESTING_DIR, &testing_files);

	Train();
	Test();

	cudaSafeCall(cudaFree(devSOM));
	return 0;
}

