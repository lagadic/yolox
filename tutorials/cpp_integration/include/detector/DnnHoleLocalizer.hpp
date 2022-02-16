#pragma once

// System
#include <memory>
#include <optional>

// Forward declaration
namespace data
{
struct DetectedHole2D;
} // namespace data

namespace detector
{

/*!
 * \class DnnHoleLocalizer.
 * \brief Deep-learning-based hole localizer.
 */
class DnnHoleLocalizer
{
public:
  DnnHoleLocalizer( const std::string &model_path, int net_size = 416, double score_threshold = 0.5 );

  ~DnnHoleLocalizer()                          = default;
  DnnHoleLocalizer( const DnnHoleLocalizer & ) = default;
  DnnHoleLocalizer( DnnHoleLocalizer && )      = default;
  DnnHoleLocalizer &operator=( const DnnHoleLocalizer & ) = default;
  DnnHoleLocalizer &operator=( DnnHoleLocalizer && ) = default;

public:
  std::vector< data::DetectedHole2D > detect( const cv::Mat &img ) const;

private:
  void init( const std::string &model_path );
  std::optional< std::vector< data::DetectedHole2D > > predict( const cv::Mat &img ) const;

private:
  int m_net_size;
  double m_score_threshold;

  struct LocalizerImpl;
  std::shared_ptr< LocalizerImpl > m_localizer_impl;
};

} // namespace detector
