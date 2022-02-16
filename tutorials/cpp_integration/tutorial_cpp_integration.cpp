// OpenCV
#include <opencv2/opencv.hpp>

// Internal
#include <data/Hole.hpp>
#include <detector/DnnHoleLocalizer.hpp>

int
main( int argc, char **argv )
{
  std::string model_path{ "" }, img_path{ "" };
  for ( auto i = 1; i < argc; i++ )
  {
    if ( std::string( argv[i] ) == "--model" )
    {
      model_path = std::string( argv[++i] );
    }
    else if ( std::string( argv[i] ) == "--img" )
    {
      img_path = std::string( argv[++i] );
    }
  }

  const auto hole_localizer = std::make_unique< detector::DnnHoleLocalizer >( model_path );

  auto cv_color = cv::imread( img_path );

  const auto detections = hole_localizer->detect( cv_color );
  std::for_each( begin( detections ), end( detections ),
                 [&cv_color]( const auto &detection ) { detection.display( cv_color ); } );

  cv::imshow( "Hole detection", cv_color );
  cv::waitKey();

  return EXIT_SUCCESS;
}
