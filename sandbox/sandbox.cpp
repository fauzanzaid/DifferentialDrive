#include <iostream>
#include <unistd.h>
#include <string>
#include <time.h>

#include <boost/thread.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>

#include "aprilvideointerface.h"
#include "pathplanners.h"
#include "controllers.h"
// For Arduino: serial port access class
#include "Serial.h"

using namespace std;
using namespace cv;

int reach_distance = 2.5;//cm
struct bot_config{
  PathPlannerGrid plan;
  PurePursuitController control;//constructor called, thus must have a default constructor with no arguments
  int id;
  robot_pose pose;
  //using intializer list allows intializing variables with non trivial contructor
  //assignment would not help if there is no default contructor with no arguments
  bot_config(int cx,int cy, int thresh,vector<vector<nd> > &tp, double a,double b,double c, int d,int e,int f,bool g):plan(PathPlannerGrid(cx,cy,thresh,tp)),control(PurePursuitController(a,b,c,d,e,f,g)){
    id = -1;//don't forget to set the id
    //below line would first call plan PathPlannerGrid constructor with no argument and then replace lhs variables with rhs ones
    //plan = PathPlannerGrid(cx,cy,thresh,tp);
  }
  void init(){
    plan.start_grid_x = plan.start_grid_y = -1;
    plan.robot_id = -1;
  }
};

int check_deadlock(vector<bot_config> &bots, int index)
{
  cout<<"\nChecking deadlock presence:\n"<<endl;
  for(int i = 0; i < bots.size(); i++)
  {
    bots[i].plan.deadlock_check_counter = 0;
  }
  int clear_flag = 0;
  int target_cell_bot_id = -1;
  while(!clear_flag)
  {
    cout<<"index: "<<index<<endl;
    int r = bots[index].plan.target_grid_cell.first;
    int c = bots[index].plan.target_grid_cell.second;
    cout<<"r,c :"<<r<<" "<<c<<endl;
    if(bots[index].plan.world_grid[r][c].bot_presence.first == 1 && bots[index].plan.world_grid[r][c].bot_presence.second != bots[index].plan.robot_tag_id)
    {
      target_cell_bot_id = bots[index].plan.world_grid[r][c].bot_presence.second;
      bots[target_cell_bot_id].plan.deadlock_check_counter++;
      if(bots[target_cell_bot_id].plan.deadlock_check_counter > 1)
      {
        break;
      }
      else if(bots[target_cell_bot_id].plan.status == 2)// to check if the said target bot has covered all its point and is in no position to move
      {
        break;
      }
      index = target_cell_bot_id;
      continue;
    }
    else
    {
      clear_flag = 1;
    }

  }
  if(clear_flag == 1)
  {
    return -1;
  }
  else
  {
    return target_cell_bot_id;
  }

}

void check_collision_possibility(AprilInterfaceAndVideoCapture &testbed, vector<PathPlannerGrid> &planners, vector<bot_config> &bots, pair<int,int> &wheel_velocities, int i)
{
  cout<<"Checking bot collision possible!\n";
  if(bots[i].plan.next_target_index != bots[i].plan.path_points.size()) //for collision avoidance
  {
    int c = (bots[i].plan.pixel_path_points[bots[i].plan.next_target_index].first)/(bots[i].plan.cell_size_x);
    int r = (bots[i].plan.pixel_path_points[bots[i].plan.next_target_index].second)/(bots[i].plan.cell_size_y);
    bots[i].plan.target_grid_cell = make_pair(r, c);
    if(bots[i].plan.world_grid[r][c].bot_presence.first == 1 && bots[i].plan.world_grid[r][c].bot_presence.second != bots[i].plan.robot_tag_id)
    {
      wheel_velocities = make_pair(0,0);
      cout<<"A Bot is present in target cell!\n";
      cout<<"r,c: "<<r<<" "<<c<<endl;
      cout<<"Present robot tag id: "<<bots[i].plan.world_grid[r][c].bot_presence.second<<endl;
      int deadlocked_bot = check_deadlock(bots, i);
      if(deadlocked_bot != -1)
      {
      cout<<"\n******\n";
      cout<<"Deadlock Detected!"<<endl;
      bots[deadlocked_bot].plan.DeadlockReplan(testbed, planners);
      cout<<"Path Replanned!"<<endl;
      cout<<"******\n\n";
      }
    }
  }
}







class ThreadSafeMat
{
  public:
    ThreadSafeMat(){};
    // {
    //  img = cv::Mat();
    // }
    ~ThreadSafeMat(){};

    void write(cv::Mat img_new){
      boost::unique_lock< boost::shared_mutex > img_mutex_lock(img_mutex);
      img = img_new;
    }

    cv::Mat read(){
      boost::shared_lock< boost::shared_mutex > img_mutex_lock(img_mutex);
      return img;
    }
  
  private:
    cv::Mat img;
    mutable boost::shared_mutex img_mutex;
};

class TSDetections
{
  public:
    TSDetections()
    {}
    
    ~TSDetections()
    {}

    void write( std::map< int, AprilTags::TagDetection > coods_new ){
      boost::unique_lock< boost::shared_mutex > coods_mutex_lock(coods_mutex);
      coods = coods_new;
    }

    std::map< int, AprilTags::TagDetection > read(){
      boost::shared_lock< boost::shared_mutex > coods_mutex_lock(coods_mutex);
      return coods;
    }

  private:
    // std::vector< AprilTags::TagDetection > coods;
    std::map< int, AprilTags::TagDetection > coods;
    mutable boost::shared_mutex coods_mutex;
};

int calculate_homographies(vector< boost::shared_ptr<ThreadSafeMat> > ts_img_ptrs, int anchor_tag_id, vector<Mat> &homographies, vector<Eigen::Matrix3d> homographies2d, vector<Mat> &img_warped_masks);
void compose_imgs(vector< boost::shared_ptr<ThreadSafeMat> > ts_img_ptrs, vector<Mat> &homographies, vector<Mat> &img_warped_masks, Mat &composition);
void capture(int device, boost::shared_ptr<ThreadSafeMat> ts_img_ptr);
void detector(int device, boost::shared_ptr<ThreadSafeMat> ts_img_ptr, boost::shared_ptr<TSDetections> ts_detections_ptr);
void transform_tag(AprilTags::TagDetection &detection, cv::Mat &homography, cv::Mat &homography_base);
// This function takes in the detections from all cameras, and outputs tag detections which are transformed to correspond to final composed image
std::vector< AprilTags::TagDetection > get_unified_detections(vector< boost::shared_ptr<TSDetections> > ts_detections_ptrs, vector<int> &active_cams, vector<int> &tag_ids, vector<Mat> &homographies);



int main(int argc, char* argv[]) {

  // Set number of cameras
  // if(argc != 2){
  //   cout << "Enter number of cameras" << endl;
  //   return -1;
  // }
  // int num_cams = stoi(argv[1]);
  int num_cams = 2;

  // Create Thread safe Mats
  vector< boost::shared_ptr<ThreadSafeMat> > ts_img_ptrs;
  for (int i = 0; i < num_cams; ++i){
    boost::shared_ptr<ThreadSafeMat> ts_img_ptr(new ThreadSafeMat);
    ts_img_ptrs.push_back( ts_img_ptr );
  }

  // Take one image from each device 
  for (int i = 0; i < num_cams; ++i){
    Mat img;
    VideoCapture vid_cap = VideoCapture(i);
    if(vid_cap.isOpened() == false)
    {
      cout << "Cannot open camera " << i << endl;
      return -1;
    }
    // namedWindow("Camera "+to_string(i),1);
    vid_cap >> img;
    vid_cap.release();

    ts_img_ptrs[i].get()->write(img);
  }

  // Calculate homographies
  vector<Mat> homographies(num_cams);
  vector<Eigen::Matrix3d> homographies2d(num_cams);
  vector<Mat> img_warped_masks(num_cams);
  int anchor_tag_id = 0;

  int status = calculate_homographies(ts_img_ptrs, anchor_tag_id, homographies, homographies2d, img_warped_masks);
  if(status == -1)
    return -1;

  // Tags to track
  vector<int> tag_ids = {0,1,2,3,4,5,6};
  int num_tags = tag_ids.size();

  // Store detections of each tag for each cam
  vector< boost::shared_ptr<TSDetections> > ts_detections_ptrs;
  for (int i = 0; i < num_cams; ++i){
    boost::shared_ptr<TSDetections> ts_detection_ptr(new TSDetections());
    ts_detections_ptrs.push_back(ts_detection_ptr);
  }

  // Store which camera is currently active for particular tag
  vector<int> active_cams(num_tags, -1);

  // Store the threads
  vector<boost::thread *> capture_threads;
  vector<boost::thread *> detector_threads;

  // Create detector and capture threads
  for (int i = 0; i < num_cams; ++i){
    capture_threads.push_back(new boost::thread(capture, i, ts_img_ptrs[i]) );
    detector_threads.push_back(new boost::thread(detector, i, ts_img_ptrs[i], ts_detections_ptrs[i]));
  }

  // Store the composed image
  // in Mat image









  AprilInterfaceAndVideoCapture testbed;
  testbed.parseOptions(argc, argv);
  // No image capture done by testbed
  // testbed.setup();
  // if (!testbed.isVideo()) {
  //   cout << "Processing image: option is not supported" << endl;
  //   testbed.loadImages();
  //   return 0;
  // }
  // cout << "Processing video" << endl;
  // testbed.setupVideo();
  // No image capture done by testbed
  int frame = 0;
  int first_iter = 1;
  double last_t = tic();
  const char *windowName = "What do you see?";
  cv::namedWindow(windowName,WINDOW_NORMAL);
  /*vector<Serial> s_transmit(2);
  ostringstream sout;
  if(testbed.m_arduino){
    for(int i = 0;i<s_transmit.size();i++){
      sout.str("");
      sout.clear();
      sout<<"/dev/ttyUSB"<<i;
      s_transmit[i].open(sout.str(),9600);
    }
  }*/
  Serial s_transmit;
  s_transmit.open("/dev/ttyUSB0", 9600);
  // No image capture done by testbed
  cv::Mat image;
  cv::Mat image_gray;
  
  //make sure that lookahead always contain atleast the next path point
  //if not then the next point to the closest would automatically become target
  //PurePursuitController controller(40.0,2.0,14.5,70,70,128,false);
  //PurePursuitController controller(20.0,2.0,14.5,70,70,128,true);
  //PathPlannerUser path_planner(&testbed);
  //setMouseCallback(windowName, path_planner.CallBackFunc, &path_planner);
  int robotCount;
  int max_robots = 5;
  



  int origin_tag_id = anchor_tag_id;//always 0
  //tag id should also not go beyond max_robots
  vector<vector<nd> > tp;//a map that would be shared among all
  vector<bot_config> bots(max_robots,bot_config(53,53,115,tp,40.0,reach_distance,14.5,75,75,128,false));
  vector<PathPlannerGrid> planners(max_robots,PathPlannerGrid(tp));

  int algo_select;
  cout<<"\nSelect the algorithm: \n" 
  "1: BSA-CM (Basic)\n" 
  "2: BSA-CM with updated Backtrack Search\n" 
  "3: Boustrophedon Motion With Updated Bactrack Search\n"
  "4: Boustrophedon Motion With BSA_CM like Backtracking\n"
  //"5: Voronoi Partition Based Online Coverage\n"
  "\nEnter here: ";
  cin>>algo_select;

/* float fx  = 5.2131891565202363e+02;
 float cx = 320;
 float fy = 5.2131891565202363e+02;
 float cy = 240; 
 Mat cameraMatrix = (Mat1d(3, 3) << fx, 0., cx, 0., fy, cy, 0., 0., 1.);

 float k1 = 1.2185003707738498e-01;
 float k2 = -2.9284657749369847e-01;
 float p1 = 0.;
 float p2 = 0. ;
 float k3 = 1.3015059691615408e-01;

 Mat distortionCoefficients = (Mat1d(1, 5) << k1, k2, p1, p2, k3);
 Mat image2;*/
 //cv::namedWindow("Original",WINDOW_NORMAL);
  while (true){
    for(int i = 0;i<max_robots;i++){
      bots[i].init();
      bots[i].id = i;//0 is saved for origin
    }
    robotCount = 0;

    // Get tag detections and display composed image
    std::vector< AprilTags::TagDetection > tag_detections = get_unified_detections(ts_detections_ptrs, active_cams, tag_ids, homographies);
    compose_imgs(ts_img_ptrs, homographies, img_warped_masks, image);
    imshow("Composition", image);
    waitKey(1);

    // Get gray composed image
    cvtColor(image, image_gray, CV_BGR2GRAY);


    // Image capture and tag detection no longer done by testbed
    // testbed.m_cap >> image;
    //image = imread("tagimage.jpg");
    // testbed.processImage(image, image_gray);//tags extracted and stored in class variable
    
    // Supply detections to testbed
    testbed.processDetections(tag_detections);

    int n = testbed.detections.size();
    for(int i = 0;i<bots.size();i++){
      bots[i].plan.robot_tag_id = i;
    }
   
    for(int i = 0;i<n;i++){     
      bots[testbed.detections[i].id].plan.robot_id = i; //robot_id is the index in detections the tag is detected at
      if(testbed.detections[i].id == origin_tag_id){//plane extracted
        bots[testbed.detections[i].id].plan.robot_id = i;
        testbed.extractPlane(i);
        //break;
      } 
    }

    if(bots[origin_tag_id].plan.robot_id<0)
      continue;//can't find the origin tag to extract plane
    for(int i = 0;i<n;i++){
      if(testbed.detections[i].id != origin_tag_id){//robot or goal
        if(robotCount>=10){
          cout<<"too many robots found"<<endl;
          break;
        }
        robotCount++;
        testbed.findRobotPose(i,bots[testbed.detections[i].id].pose);//i is the index in detections for which to find pose
      }
    }

    /*cout<<"************\n";
    for(int i = 0; i < n; i++)
    {
      if(testbed.detections[i].id > 4) continue;
      cout<<testbed.detections[i].id<<" ";
      testbed.findRobotPose(i,bots[testbed.detections[i].id].pose);
      cout<<"pose: "<<bots[testbed.detections[i].id].plan.robot_tag_id<<" "<<bots[testbed.detections[i].id].pose.x<<" "<<bots[testbed.detections[i].id].pose.y<<" "<<bots[testbed.detections[i].id].pose.omega<<endl;

      cout<<"detection id: "<<testbed.detections[i].id<<endl;
      cout<<"robot_tag_id vs robot_id:    "<< bots[testbed.detections[i].id].plan.robot_tag_id<<" "<<bots[testbed.detections[i].id].plan.robot_id<<endl;
      cout<<"pose: "<<bots[testbed.detections[i].id].plan.robot_tag_id<<" "<<bots[testbed.detections[i].id].pose.x<<" "<<bots[testbed.detections[i].id].pose.y<<" "<<bots[testbed.detections[i].id].pose.omega<<endl;
      cout<<"func: "<<bots[testbed.detections[i].id].plan.robot_id<<" "<<x<<" "<<y<<endl;

    }
    cout<<"*********\n";*/
    //all robots must be detected(in frame) when overlay grid is called else some regions on which a robot is 
    //present(but not detected) would be considered an obstacle
    //no two robots must be present in the same grid cell(result is undefined)
    if(first_iter){
      //first_iter = 0; 
      bots[0].plan.overlayGrid(testbed.detections,image_gray);//overlay grid completely reintialize the grid, we have to call it once at the beginning only when all robots first seen simultaneously(the surrounding is assumed to be static) not every iteration
      for(int i = 1;i<bots.size();i++){
        bots[i].plan.rcells = bots[0].plan.rcells;
        bots[i].plan.ccells = bots[0].plan.ccells;
      }
    }

    //the planners[i] should be redefined every iteration as the stack and bt points change 
    //this is inefficient, should look for alternates
    for(int i = 0;i<bots.size();i++){
      //for bot 0, the origin and robot index would be the same
      bots[i].plan.origin_id = bots[0].plan.robot_id;//set origin index of every path planner which is the index of tag 0 in detections vector given by RHS
      planners[i] = bots[i].plan;
    }

    if(first_iter)
    {
    	first_iter = 0;
    	if(algo_select==5)
    	{
    		bots[0].plan.defineVoronoiPartition(testbed, planners);
    	}    	
    }

    for(int i = 1;i<bots.size();i++){
      cout<<"planning for id "<<i<<endl;
      switch(algo_select)
      {
      case 1: bots[i].plan.BSACoverageIncremental(testbed,bots[i].pose, reach_distance,planners); break;
      case 2: bots[i].plan.BSACoverageWithUpdatedBactrackSelection(testbed,bots[i].pose, reach_distance,planners); break;
      case 3: bots[i].plan.BoustrophedonMotionWithUpdatedBactrackSelection(testbed,bots[i].pose, reach_distance,planners); break;
      case 4: bots[i].plan.BoustrophedonMotionWithBSA_CMlikeBacktracking(testbed,bots[i].pose, reach_distance,planners); break;
      case 5: bots[i].plan.VoronoiPartitionBasedOnlineCoverage(testbed,bots[i].pose, reach_distance,planners); break;
      default: bots[i].plan.BSACoverageIncremental(testbed,bots[i].pose, reach_distance,planners);   
      }   
    }


    if(testbed.m_arduino){
      pair<int,int> wheel_velocities;
      for(int i = 1;i<bots.size();i++){//0 is for origin

        int next_point_index_in_path=0; //for collision avoidance
        cout<<i<<": robot pose: "<<bots[i].pose.x<<" "<<bots[i].pose.y<<" "<<bots[i].pose.omega<<endl;
        wheel_velocities = bots[i].control.computeStimuli(bots[i].pose,bots[i].plan.path_points, next_point_index_in_path);//for nonexistent robots, path_points vector would be empty thus preventing the controller to have any effect
          
          bots[i].plan.next_target_index = next_point_index_in_path;
          check_collision_possibility(testbed, planners, bots, wheel_velocities, i);
          
          
         /* s_transmit[i-1].print((unsigned char)(bots[i].id));
          cout<<"sending velocity for bot "<<bots[i].id<<endl;
          s_transmit[i-1].print((unsigned char)(128+wheel_velocities.first));
          s_transmit[i-1].print((unsigned char)(128+wheel_velocities.second));
        */
          s_transmit.print((unsigned char)(bots[i].id));
          cout<<"sending velocity for bot "<<bots[i].id<<endl;
          s_transmit.print((unsigned char)(128+wheel_velocities.first));
          s_transmit.print((unsigned char)(128+wheel_velocities.second));
          cout<<"sent velocities "<<wheel_velocities.first<<" "<<wheel_velocities.second<<endl;

          
      }
    }
    if(testbed.m_draw){
      for(int i = 0;i<n;i++){
        testbed.detections[i].draw(image);
      }
      bots[origin_tag_id].plan.drawGrid(image, planners);
      for(int i = 1;i<bots.size();i++){
      	//bots[i].plan = planners[i];
        bots[i].plan.drawPath(image);
      }
      //add a next point circle draw for visualisation
      //add a only shortest path invocation drawing function in pathplanners
      //correct next point by index to consider reach radius to determine the next point
      imshow(windowName,image);
      //imshow("Original", image2);
    }
    // print out the frame rate at which image frames are being processed
    frame++;

    if (frame % 10 == 0) {
      double t = tic();
      //cout<<"image size: "<<image.cols<<"x"<<image.rows<<endl;
      cout<<"************************\n";
      cout << "  " << 10./(t-last_t) << " fps" << endl;
      cout<<"************************\n";
      last_t = t;
    }
    if (cv::waitKey(10) == 27){
      if(testbed.m_arduino){
        for(int i = 1;i<bots.size();i++){//0 is for origin
          //sif (i==1)continue;
           /* s_transmit[i-1].print((unsigned char)(bots[i].id));
            s_transmit[i-1].print((unsigned char)(0));
            s_transmit[i-1].print((unsigned char)(0));*/
            s_transmit.print((unsigned char)(bots[i].id));
            s_transmit.print((unsigned char)(0));
            s_transmit.print((unsigned char)(0));
            //break;
        }
      }
      break;//until escape is pressed
    }
  }


  // Destroy threads
  for (int i = 0; i < num_cams; ++i){
    capture_threads[i]->join();
    delete capture_threads[i];

    detector_threads[i]->join();
    delete detector_threads[i];
  }


  return 0;
}








int calculate_homographies(vector< boost::shared_ptr<ThreadSafeMat> > ts_img_ptrs, int anchor_tag_id, vector<Mat> &homographies, vector<Eigen::Matrix3d> homographies2d, vector<Mat> &img_warped_masks){
  int num_imgs = ts_img_ptrs.size();

  vector<Mat> imgs_gray(num_imgs);

  AprilTags::TagCodes tagCodes = AprilTags::tagCodes36h11;
  AprilTags::TagDetector *tagDetector = new AprilTags::TagDetector(tagCodes);

  vector<vector<AprilTags::TagDetection>> tag_detection_vectors(num_imgs);
  vector<vector<Point2f>> ref_points_vectors;
  vector<AprilTags::TagDetection> anchor_tag_detections(num_imgs);
  

  // Detect reference points through anchor tag
  for (int i = 0; i < num_imgs; ++i)
  {
    cvtColor(ts_img_ptrs[i].get()->read(), imgs_gray[i], CV_BGR2GRAY);
    tag_detection_vectors[i] = tagDetector->extractTags(imgs_gray[i]);

    // cout << "Camera " << i << endl;
    bool anchor_tag_found(false);
    for (int j = 0; j < tag_detection_vectors[i].size(); ++j)
    {
      // cout << "\t" << tag_detection_vectors[i][j].id << "\t" << tag_detection_vectors[i][j].cxy.first << " "<< tag_detection_vectors[i][j].cxy.second << endl;
      
      if(tag_detection_vectors[i][j].id == anchor_tag_id){

        anchor_tag_detections[i] = tag_detection_vectors[i][j];

        // vector<Point2f> ref_points(4);
        vector<Point2f> ref_points;
        for (int k = 0; k < 4; ++k)
        {
          ref_points.push_back(Point2f(tag_detection_vectors[i][j].p[k].first, tag_detection_vectors[i][j].p[k].second));
          cout << i << " " << k << " " << tag_detection_vectors[i][j].p[k].first << " " << tag_detection_vectors[i][j].p[k].second << endl;
        }
        ref_points.push_back(Point2f(tag_detection_vectors[i][j].cxy.first, tag_detection_vectors[i][j].cxy.second));
        cout << i << " c" << " " << tag_detection_vectors[i][j].cxy.first << " " << tag_detection_vectors[i][j].cxy.second << endl;
        ref_points_vectors.push_back(ref_points);

        anchor_tag_found = true;

      }
    }

    if(anchor_tag_found == false){
      cout << "Unable to find tag id " << anchor_tag_id << endl;
      return -1;
    }
  }


  // Calculate homographies
  for (int i = 0; i < num_imgs; ++i)
  {
    homographies[i] = findHomography(ref_points_vectors[i], ref_points_vectors[0]);
    // view 0 is taken as the base
    // cout << hg << endl << endl;
  }

  // Stroe homographies2d
  for (int i = 0; i < num_imgs; ++i)
  {
    homographies2d[i] = anchor_tag_detections[i].homography;
    // Eigen::Matrix<double,3,3> m = anchor_tag_detections[i].homography;

  }


  // Calculate extreme points
  vector<Point2f> img_corners;
  img_corners.push_back(Point2f(0,0));
  img_corners.push_back(Point2f(0,480));
  img_corners.push_back(Point2f(640,480));
  img_corners.push_back(Point2f(640,0));
  // x=w y=h
  double w_min=10000, w_max=-10000, h_min=10000, h_max=-10000; // Careful!
  vector<vector<Point2f>> imgs_warped_corners(num_imgs);
  for (int i = 0; i < num_imgs; ++i)
  {
    perspectiveTransform(img_corners, imgs_warped_corners[i], homographies[i]);

    for (int j = 0; j < 4; ++j)
    {
      cout << i << " " << j << " " << imgs_warped_corners[i][j] << endl;
      if(imgs_warped_corners[i][j].x < w_min) w_min = imgs_warped_corners[i][j].x;
      if(imgs_warped_corners[i][j].x > w_max) w_max = imgs_warped_corners[i][j].x;
      if(imgs_warped_corners[i][j].y < h_min) h_min = imgs_warped_corners[i][j].y;
      if(imgs_warped_corners[i][j].y > h_max) h_max = imgs_warped_corners[i][j].y;
    }
  }
  // cout << " " << w_min << " " << w_max << " " << h_min << " " << h_max << endl;

  int output_w = ceil(w_max-w_min);
  int output_h = ceil(h_max-h_min);

  double shift_w = -w_min;
  double shift_h = -h_min;

  Mat homography_shift = (Mat_<double>(3, 3) << 1,0,shift_w,0,1,shift_h,0,0,1);
  // Eigen::Matrix3d homography2d_shift = (Eigen::Matrix4d() << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16).finished();

  // Add translation to homographies
  for (int i = 0; i < num_imgs; ++i)
  {
    homographies[i] = homography_shift*homographies[i];
  }

  // Create ROI masks
  Mat mask_base = Mat(480, 640, CV_8UC1);
  mask_base.setTo(cv::Scalar(255,255,255));
  Mat output_covered = Mat(output_h, output_w, CV_8UC1);
  output_covered.setTo(cv::Scalar(0));

  for (int i = 0; i < num_imgs; ++i)
  {
    warpPerspective(mask_base, img_warped_masks[i], homographies[i], Size(output_w, output_h));
    // Add uncovered area only
    img_warped_masks[i] -= output_covered;
    output_covered += img_warped_masks[i];
  }

  return 0;
}

void compose_imgs(vector< boost::shared_ptr<ThreadSafeMat> > ts_img_ptrs, vector<Mat> &homographies, vector<Mat> &img_warped_masks, Mat &composition){
  Mat img_warped;
  composition = Mat(img_warped_masks[0].rows, img_warped_masks[0].cols, CV_64F, double(5));
  for (int i = 0; i < ts_img_ptrs.size(); ++i)
  {
    warpPerspective(ts_img_ptrs[i].get()->read(), img_warped, homographies[i], img_warped_masks[0].size());
    img_warped.copyTo(composition, img_warped_masks[i]);
  }
}

void capture(int device, boost::shared_ptr<ThreadSafeMat> ts_img_ptr){

  cv::VideoCapture vid_cap = cv::VideoCapture(device);
  cv::Mat img;

  int index = 0;
  double c0=getTickCount(), c1;

  while(1)
  // for (int i = 0; i < 100; ++i)
  {
    vid_cap >> img;
    ts_img_ptr.get()->write(img);

    index++;
    if(index%100 == 0){
      c1=getTickCount();
      // cout << "capture : "<<device<<" fps " << getTickFrequency() * 100.0 / (c1-c0) << endl;
      c0 = c1;
    }
  }
}

void detector(int device, boost::shared_ptr<ThreadSafeMat> ts_img_ptr, boost::shared_ptr<TSDetections> ts_detections_ptr){
  cv::Mat img;
  cv::Mat img_gray;


  AprilTags::TagCodes tagCodes = AprilTags::tagCodes36h11;
  AprilTags::TagDetector *tagDetector = new AprilTags::TagDetector(tagCodes);
  vector<AprilTags::TagDetection> tag_detections;


  int index = 0;
  double c0=getTickCount(), c1;

  while(1)
  // for (int i = 0; i < 100; ++i)
  {
    img = ts_img_ptr.get()->read();
    
    cvtColor(img, img_gray, CV_BGR2GRAY);
    tag_detections = tagDetector->extractTags(img_gray);



    // std::vector< AprilTags::TagDetection > coods;
    std::map< int, AprilTags::TagDetection > coods;

    for (int i = 0; i < tag_detections.size(); ++i)
    {
      // AprilTags::TagDetection cood = tag_detections[i].cxy;
      coods[ tag_detections[i].id ] = tag_detections[i];
      // cout << "camera:" << device << " tag:" << tag_detections[i].id << endl;
    }

    ts_detections_ptr.get()->write(coods);


    index++;
    if(index%100 == 0){
      c1=getTickCount();
      // cout << "detector : "<<device<<" fps " << getTickFrequency() * 100.0 / (c1-c0) << endl;
      c0 = c1;
    }
  }
}

std::vector< AprilTags::TagDetection > get_unified_detections(vector< boost::shared_ptr<TSDetections> > ts_detections_ptrs, vector<int> &active_cams, vector<int> &tag_ids, vector<Mat> &homographies){
  
  std::vector< AprilTags::TagDetection > unified_tag_detections;

  vector< std::map< int,AprilTags::TagDetection > > cam_tag_detections;
  for (int i = 0; i < ts_detections_ptrs.size(); ++i)
  {
    // For each camera
    cam_tag_detections.push_back( ts_detections_ptrs[i].get()->read() );
  }

  for (int i = 0; i < tag_ids.size(); ++i)
  {
    // For each tag which has an active camera
    if(active_cams[i] != -1){
      try{
        // Get value if it exists in the tags detected by this tag's active camera
        AprilTags::TagDetection detection = cam_tag_detections[ active_cams[i] ].at(tag_ids[i]);

        // Transform
        transform_tag(detection, homographies[ active_cams[i] ], homographies[0]);

        // Add to map
        unified_tag_detections.push_back(detection);

        // cout << "active same  tag:" << tag_ids[i] << " camera:" << active_cams[i] << endl;
      }
      catch(const std::out_of_range &oorerr){
        // Value does not exist, need to search for another camera
        active_cams[i] = -1;
      }
    }
  }

  for (int i = 0; i < tag_ids.size(); ++i){
    // For each tag which does not have an active camera
    if(active_cams[i] == -1){

      for (int j = 0; j < ts_detections_ptrs.size(); ++j)
      {
        // Search in each camera
        try{
          // Get value if it exists in the tags detected by camera j
          AprilTags::TagDetection detection = cam_tag_detections[ j ].at(tag_ids[i]);
          active_cams[i] = j;
          
          // Transform
          transform_tag(detection, homographies[ active_cams[i] ], homographies[0]);

          // Add to map
          unified_tag_detections.push_back(detection);

          // cout << "active new   tag:" << tag_ids[i] << " camera:" << active_cams[i] << endl;

          // Camera found
          break;
        }
        catch(const std::out_of_range &oorerr){
          // Value does not exist, need to search for another camera
          active_cams[i] = -1;
        }

        if(active_cams[i] == -1){
          // cout << "active none  tag:" << tag_ids[i] << " camera:" << active_cams[i] << endl;
        }
      }
    }
  }

  return unified_tag_detections;
}

void transform_tag(AprilTags::TagDetection &detection, cv::Mat &homography, cv::Mat &homography_base){

  // !! observecPerimeter not transformed

  /////////////////////////
  // Transform p and cxy //
  /////////////////////////

  vector<Point2f> points_old;
  vector<Point2f> points_new;

  // Get old coods
  for (int i = 0; i < 4; ++i)
  {
    points_old.push_back(Point2f(detection.p[i].first, detection.p[i].second));
  }
  points_old.push_back(Point2f(detection.cxy.first, detection.cxy.second));

  // Calculate new coods
  perspectiveTransform(points_old, points_new, homography);

  // Assign new coods
  for (int i = 0; i < 4; ++i)
  {
    detection.p[i].first = points_new[i].x;
    detection.p[i].second = points_new[i].y;
  }

  detection.cxy.first = points_new[4].x;
  detection.cxy.second = points_new[4].y;



  //////////////////////////
  // Transform homography //
  //////////////////////////

  Eigen::Matrix3d homography_eigen, homography_base_eigen;
  cv2eigen(homography, homography_eigen);
  cv2eigen(homography_base, homography_base_eigen);

  // cout << endl;
  // cout << detection.id << endl;
  // cout << detection.homography << endl;
  // cout << endl;

  detection.homography = homography_eigen * homography_base_eigen.inverse() * detection.homography;

  // cout << detection.homography << endl;
  // cout << endl;

  ///////////////////
  // Transform hxy //
  ///////////////////
  // Not done. (Not needed?)
}

