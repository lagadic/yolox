#include <data/Hole.hpp>

// System
#include <iomanip>

// OpenCV
#include <opencv2/imgproc.hpp>

namespace data
{

DetectedHole2D::DetectedHole2D( int u_min, int u_max, int v_min, int v_max, unsigned int cls, double score )
  : bbox{ cv::Point{ u_min, v_min }, cv::Point{ u_max, v_max } }
  , cls{ cls }
  , score{ score } {};

/*!
 * Display the detected hole (ie, bbox and score) in an image.
 *
 * \param[in] img : Image used as background.
 * \param[in] color : Color used to draw the bbox.
 */
void
DetectedHole2D::display( cv::Mat &img, const cv::Scalar &color ) const
{
  cv::rectangle( img, bbox.tl(), bbox.br(), color );

  std::stringstream ss;
  ss << cls << " | " << std::setprecision( 4 ) << score * 100 << "%";
  cv::putText( img, ss.str(), bbox.tl(), cv::FONT_HERSHEY_DUPLEX, 0.5, color );
}

} // namespace data
