#pragma once

// OpenCV
#include <opencv2/core/types.hpp>

namespace data
{

/*!
 * \struct DetectedHole2D
 *
 * 2D Hole data (specialized for machine learning detection and localisation).
 */
struct DetectedHole2D
{
  virtual ~DetectedHole2D()                = default;
  DetectedHole2D( const DetectedHole2D & ) = default;
  DetectedHole2D( DetectedHole2D && )      = default;
  DetectedHole2D &operator=( const DetectedHole2D & ) = default;
  DetectedHole2D &operator=( DetectedHole2D && ) = default;

  explicit DetectedHole2D( int u_min, int u_max, int v_min, int v_max, unsigned int cls, double score );

  void display( cv::Mat &img, const cv::Scalar &color = { 0, 255, 0 } ) const;

  cv::Rect bbox;
  unsigned int cls;
  double score;
};

} // namespace data
