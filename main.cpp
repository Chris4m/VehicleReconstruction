#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <strstream>
#include <vector>
#include <string>
#include <filesystem>

#include <opencv2\core.hpp> 

#include <MaskRCNN.h>
#include <VehicleKeypoint.h>

using namespace std;
using namespace cv;

vector<string> split(string str, char delimiter) {
	vector<string> internal;
	stringstream ss(str); // Turn the string into a stream.
	string tok;

	while (getline(ss, tok, delimiter)) {
		internal.push_back(tok);
	}

	return internal;
}

void write3DToFile(Mat img3D, String outputCoords, Mat img) {
	ofstream output3DCloud;
	output3DCloud.open(outputCoords);
	
	Size s = img3D.size();
	for (int i = 0; i < s.height; i++) {
		for (int j = 0; j < s.width; j++) {
			Vec3b color = img.at<Vec3b>(i, j);
			float x = img3D.at<Point3f>(i, j).x;
			float y = img3D.at<Point3f>(i, j).y;
			float z = img3D.at<Point3f>(i, j).z;

			if (z <= 30) {
				output3DCloud << x << "," << y << "," << z << "," << to_string(color.val[0]) << "," << to_string(color.val[1])  << "," << to_string(color.val[2]) << endl;
			}
		}
	}
	output3DCloud.close();
}

//cv::Mat computeQMatrix(vector<string> mat_P2, vector<string> mat_P3) {
//	// P2
//	float cx_P2 = stof(mat_P2[3]);
//	float cy_P2 = stof(mat_P2[7]);
//	float f_P2 = stof(mat_P2[1]);
//	// Calculate Tx
//	float tx = 0.54;
//	//float tx;
//	//tx = norm(P2-P3);
//	float valuesQ[16] = { 1.0, 0.0, 0.0, -fabs(cx_P2), 0.0, 1.0, 0.0, -fabs(cy_P2), 0.0 , 0.0 , 0.0 , -fabs(f_P2), 0.0 , 0.0 , -1.0 / tx , 0.0 };
//	
//	Mat Q = Mat(4, 4, CV_32F, valuesQ);
//	return Q.clone();
//}

// ***************************************************************************************
int main(int argc, char* argv[]) {

	// ===============================================================================================
	// EXAMPLE FOR THE USAGE OF MASK RCNN
	// ===============================================================================================
	
	// initialization for loading all the images of the folder "./Data/Sequence/images/"
	namespace fs = std::filesystem;
	std::string imageDirectory = "";
	std::string src_directory = "./Data/Sequence/images/";

	// iteration over all images
	for (const auto& entry : fs::directory_iterator(src_directory)) {
		// get image name
		imageDirectory = entry.path().string();
		std::string imageName = imageDirectory.substr(23, 6);
		
		// load images and defining output directory
		String imagePath = src_directory + imageName + ".png";
		Mat img = imread(imagePath);
		String dst_directory = "C:/Users/phili/Seafile/F&E IPI/Data/VehicleCropped_offset/";

		// initialize the Mask RCNN
		string fnMaskRCNNDefFile = "./Data/MaskRCNN/MaskRCNNDefinitionFile.txt";
		MaskRCNN maskRCNN;
		maskRCNN.init(fnMaskRCNNDefFile);

		// detect the vehicles
		int nVehicles;
		Mat imgSeg;
		vector<double> vScores;
		maskRCNN.detectCars(img, imgSeg, vScores, nVehicles, 0.7, 0.3);
		imgSeg.convertTo(imgSeg, CV_8UC3, 255.0 / nVehicles);

		// initialization including pixel offset for cropped images
		int offset = 5;
		float min_threshold = 0.05;
		int const number_keypoints = 36;
		String image_out = "";

		// checking if any vehicles have been detected
		if (nVehicles != 0) {

			// initialization for each detected image
			int noCroppedVehicle = 1;			// count the number of detected vehicles in one image
			Size imgSegSize = imgSeg.size();
			int rows = imgSegSize.height;
			int cols = imgSegSize.width;

			//+++++++++++++++++++++++++++ Car ID Mask of vehicles +++++++++++++++++++++++++++

			Mat carIdentification(Size(cols, rows), CV_8UC1, Scalar(0));

			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					uint8_t carid = imgSeg.at<uint8_t>(i, j);
					if (carid != 0) {
						carIdentification.at<uint8_t>(i, j) = carid / (255 / nVehicles);
					}
				}
			}

			// read disparity
			//String disp_directory = "./Data/Sequence/disparities/";
			//Mat disparity = imread(disp_directory + imageName + ".exr", IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
			//disparity.convertTo(disparity, CV_32FC1);
			//// projection matrix Q
			//String calib_directory = "./Data/Sequence/calib/";
			//float m;
			//ifstream fileStream(calib_directory + imageName + ".txt");
			//string line;
			//vector<string> v_split;
			//vector<string> mat_P2, mat_P3;
			//while (getline(fileStream, line)) {
			//	v_split = split(line, ' ');
			//	if (v_split[0] == "P2:") {
			//		mat_P2 = v_split;
			//		continue;
			//	}
			//	if (v_split[0] == "P3:") {
			//		mat_P3 = v_split;
			//		break;
			//	}
			//}
			//cv::Mat Q = computeQMatrix(mat_P2, mat_P3);
			//Mat img3D;
			//cv::reprojectImageTo3D(disparity, img3D, Q);
			//String pointCloud_image = "./Data/PointCloud/" + imageName + ".txt";
			//write3DToFile(img3D, pointCloud_image, img);

			//+++++++++++++++++++++++++++ RGB Mask of vehicles +++++++++++++++++++++++++++
			Vec3b tmp_vec = (0, 0, 0);

			for (int k = 1; k <= nVehicles; k++) {
				int min_rows = rows;
				int min_cols = cols;
				int max_rows = 0;
				int max_cols = 0;
				for (int i = 0; i < rows; i++) {
					for (int j = 0; j < cols; j++) {
						if (carIdentification.at<uint8_t>(i, j) == k) {
							
							// get border coords for the cropped image (car)
							if (j < min_cols) {
								min_cols = j;
							}
							if (i < min_rows) {
								min_rows = i;
							}
							if (j > max_cols) {
								max_cols = j;
							}
							if (i > max_rows) {
								max_rows = i;
							}
						}
					}
				}

				// check if X or Y are located on the edges of the image
				if (min_rows >= offset) {
					min_rows -= offset;
				} else min_rows = 0;

				if (min_cols >= offset) {
					min_cols -= offset;
				} else min_cols = 0;

				if (max_rows <= rows - offset) {
					max_rows += offset;
				} else max_rows = rows;

				if (max_cols <= cols - offset) {
					max_cols += offset;
				} else max_cols = cols;

				// initialization for cropping one image for each car
				int size_cols = max_cols - min_cols;
				int size_rows = max_rows - min_rows;
				Mat rgbMask(Size(size_cols, size_rows), CV_8UC3, tmp_vec);

				int min_idxRows = min_rows;
				int min_idxCols = min_cols;

				//ofstream output3DCloud;
				//String outputCarCloud = "./Data/PointCloud/" + imageName + "-" + to_string(noCroppedVehicle) + ".txt";
				//output3DCloud.open(outputCarCloud);

				for (int i = 0; i < size_rows; i++) {
					for (int j = 0; j < size_cols; j++) {
						// if:   "erase" other cars than the current of the image
						// else: write RGB-information in the cropped image mat
						if (carIdentification.at<uint8_t>(min_idxRows, min_idxCols) != k && carIdentification.at<uint8_t>(min_idxRows, min_idxCols) != 0) {
							rgbMask.at<Vec3b>(i, j)(0) = 255;
							rgbMask.at<Vec3b>(i, j)(1) = 255;
							rgbMask.at<Vec3b>(i, j)(2) = 255;
							min_idxCols++;
						} else {
							if (carIdentification.at<uint8_t>(min_idxRows, min_idxCols) != 0) {
								Vec3b color_glob = img.at<Vec3b>(i + min_rows, j + min_cols);
								//float x = img3D.at<Point3f>(i + min_rows, j + min_cols).x;
								//float y = img3D.at<Point3f>(i + min_rows, j + min_cols).y;
								//float z = img3D.at<Point3f>(i + min_rows, j + min_cols).z;
								//if (z <= 30) {
								//	output3DCloud << x << "," << y << "," << z << "," << to_string(color_glob.val[0]) << "," << to_string(color_glob.val[1]) << "," << to_string(color_glob.val[2]) << endl;
								//}
							}

							Vec3b color = img.at<Vec3b>(min_idxRows, min_idxCols);
							rgbMask.at<Vec3b>(i, j)(0) = color.val[0];
							rgbMask.at<Vec3b>(i, j)(1) = color.val[1];
							rgbMask.at<Vec3b>(i, j)(2) = color.val[2];
							min_idxCols++;
						}
					}
					min_idxRows++;
					min_idxCols = min_cols;
				}
				//output3DCloud.close();

				// save cropped image in destination directory
				image_out = dst_directory + imageName + "_" + to_string(noCroppedVehicle) + ".png";
				imwrite(image_out, rgbMask);

				// ===============================================================================================
				// KEYPOINT CNN
				// ===============================================================================================

				// Compare detected vehicle with label data
				String label_directory = "./Data/Sequence/label/";
				bool isCar = false;
				int line_of_car = 0;

				ifstream fileStream(label_directory + imageName + ".txt");
				string line;
				vector<string> v_split;
				vector<string> mat_Car;

				while (getline(fileStream, line)) {
					v_split = split(line, ' ');
					if (v_split[0] == "Car") {
						mat_Car = v_split;
						//cout << "Min cols: " << to_string(min_cols) << ", Min rows: " << to_string(min_rows) << ", Max cols: " << to_string(max_cols) << ", Max rows: " << to_string(max_rows) << endl;
						// Compare bounding boxes
						Rect reference_box = Rect(Point2f(stof(mat_Car[4]), stof(mat_Car[5])), Point2f(stof(mat_Car[6]), stof(mat_Car[7])));
						Rect car_box = Rect(Point2f(min_cols, min_rows), Point2f(max_cols, max_rows));

						Rect overlap = car_box & reference_box;
						//cout << "Overlap Area: " << to_string(overlap.area()) << ", Reference Area: " << to_string(reference_box.area()) << ", Car Area: " << to_string(car_box.area()) << endl;
						float diff = (float) overlap.area() / (reference_box.area() + car_box.area() - overlap.area());
						//cout << "Difference: " << to_string(diff) << endl;
						
						if (overlap.area() == car_box.area()) {
							//cout << "Fully overlap" << endl;
							isCar = true;
							line_of_car++;
							break;
						} else if (diff >= 0.7) {
							isCar = true;
							line_of_car++;
							break;
						}
					}
					line_of_car++;
				}
				
				if (!isCar) {
					line_of_car = -1;
				}

				// read image
				Mat imgKPt = imread(image_out);

				// initialize the keypoint CNN
				String fnKeypointCNNDefFile = "./Data/KeypointCNN/KeypointCNNDefinitionFile.txt";
				VehicleKeypoint keypointCNN;
				keypointCNN.init(fnKeypointCNNDefFile);

				// get Keypoint heatmaps
				bool doVisualisation = true;
				vector<Mat> vHeatmaps;
				keypointCNN.getKeypointHeatmaps(imgKPt, vHeatmaps, doVisualisation);

				// initialize keypoint extraction
				Mat probability;
				float maxProb = 0.0;
				float xProb = -1.0;
				float yProb = -1.0;
				float zProb = -1.0;

				// output file containing the coordinates of the keypoints
				ofstream keypointFile;
				String outputCoords = "C:/Users/phili/Seafile/F&E IPI/Data/ResultKP/MaxProb_2D_offset/" + imageName + "-" + to_string(noCroppedVehicle) + ".txt";
				keypointFile.open(outputCoords);

				if (!vHeatmaps.empty()) {
					float image_probability[number_keypoints] = { 0.0 };
					vector<Point2f> coords;
					// iteration over all keypoints
					for (int i = 0; i < vHeatmaps.size() - 4; i++) {
						probability = vHeatmaps[i];
						for (int j = 0; j < probability.rows; j++) {
							for (int k = 0; k < probability.cols; k++) {
								// Heatmap probs ausgeben
								float prob_tmp = probability.at<float>(j, k);
								//meanProb += prob_tmp;
								if (prob_tmp > maxProb) {
									maxProb = prob_tmp;
									yProb = j + min_rows;
									xProb = k + min_cols;
								}
							}
						}
						image_probability[i] = maxProb;
						coords.push_back(Point2f(xProb, yProb));

						xProb = -1.0;
						yProb = -1.0;
						maxProb = 0.0;
					}

					// Remove dupilcate Pixel for more than 1 KP
					for (int i = 0; i < number_keypoints - 1; i++) {
						if (coords[i].x != -1) {
							for (int j = i + 1; j < number_keypoints; j++) {
								if (coords[j].x != -1) {
									if (coords[i].x == coords[j].x and coords[i].y == coords[j].y) {
										if (image_probability[i] > image_probability[j]) {
											coords[j].x = -1;
											coords[j].y = -1;
											image_probability[j] = -1;
										}
										else {
											coords[i].x = -1;
											coords[i].y = -1;
											image_probability[i] = -1;
											break;
										}
									}
								}
							}
						}
					}

					// Mean of probabilities with threshold 0.05
					float sum_probs = 0.0;
					float mean_probs = 0.0;
					int index = 0;

					for (int i = 0; i < number_keypoints; i++) {
						if (image_probability[i] > min_threshold) {
							sum_probs += image_probability[i];
							index++;
						}
					}
					mean_probs = sum_probs / index;

					// Variance of probabilities with threshold 0.05
					float variance_probs = 0.0;
					for (int i = 0; i < number_keypoints; i++) {
						if (image_probability[i] > min_threshold) {
							variance_probs += pow(image_probability[i] - mean_probs, 2);
						}
					}
					variance_probs = variance_probs / index;

					// Standard Deviation of probabilities with threshold 0.05
					float stdDeviation_probs = 0.0;
					stdDeviation_probs = sqrt(variance_probs);
					//cout << "Mean: " << to_string(mean_probs) << ", StdDev: " << to_string(stdDeviation_probs) << endl;

					for (int i = 0; i < number_keypoints; i++) {
						if (image_probability[i] > (mean_probs - stdDeviation_probs)) {
							keypointFile << to_string(coords[i].x) << "," << to_string(coords[i].y) << "," << to_string(image_probability[i]) << "," << to_string(line_of_car) << "\n";
						}
						else {
							keypointFile << to_string(-1) << "," << to_string(-1) << "," << to_string(-1) << "," << to_string(-1) << "\n";
						}
					}

					keypointFile.close();
				}

				noCroppedVehicle++;
				outputCoords = "";
				image_out = "";
			}
		}
	}
	return 0;
}
