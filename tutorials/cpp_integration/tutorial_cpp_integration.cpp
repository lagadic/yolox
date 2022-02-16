// OpenCV
#include <opencv2/opencv.hpp>

// Internal
#include <data/Hole.hpp>
#include <detector/DnnHoleLocalizer.hpp>

int
main( int argc, char **argv )
{
  const auto model_path     = std::string( "/root/deep-learning-ws/yolox/example/hole_detector/models/tiny_yolox" );
  const auto hole_localizer = std::make_unique< detector::DnnHoleLocalizer >( model_path );

  auto cv_color = cv::imread( "/root/deep-learning-ws/yolox/example/hole_detector/data/Images/frame0000.jpg" );

  const auto detections = hole_localizer->detect( cv_color );
  std::for_each( begin( detections ), end( detections ),
                 [&cv_color]( const auto &detection ) { detection.display( cv_color ); } );

  cv::imshow( "Hole detection", cv_color );
  cv::waitKey();

  return EXIT_SUCCESS;
}
