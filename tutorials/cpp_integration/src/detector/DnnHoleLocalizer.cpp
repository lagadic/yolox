// System
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <cstdlib>

// OpenCV
#include <opencv2/imgproc.hpp>

// Internal
#include <data/Hole.hpp>
#include <detector/DnnHoleLocalizer.hpp>

// Local helpers
namespace
{

namespace py = boost::python;
namespace np = boost::python::numpy;

/*!
 * Display python exception.
 */
void
pythonException()
{
  PyObject *exc, *val, *tb;
  PyErr_Fetch( &exc, &val, &tb );
  PyErr_NormalizeException( &exc, &val, &tb );

  std::string exception_error;
  py::handle<> hexc( exc ), hval( py::allow_null( val ) ), htb( py::allow_null( tb ) );
  if ( !hval )
  {
    exception_error = py::extract< std::string >( py::str( hexc ) );
  }
  else
  {
    py::object traceback( py::import( "traceback" ) );
    py::object format_exception( traceback.attr( "format_exception" ) );
    py::object formatted_list( format_exception( hexc, hval, htb ) );
    py::object formatted( py::str( "" ).join( formatted_list ) );
    exception_error = py::extract< std::string >( formatted );
  }

  throw( std::runtime_error( "Caught python exception: " + exception_error ) );
}

/*!
 * Convert python bbox into cpp bbox.
 *
 * \param[in] boxes : Python bbox list.
 * \return Vector of detected holes (bbox, score and class).
 */
std::vector< data::DetectedHole2D >
boxes_to_vector( const py::list &boxes )
{
  std::vector< data::DetectedHole2D > holes;

  for ( int i = 0; i < py::len( boxes ); ++i )
  {
    const auto xmin = static_cast< double >( py::extract< float >( boxes[i].attr( "xmin" ) ) );
    const auto ymin = static_cast< double >( py::extract< float >( boxes[i].attr( "ymin" ) ) );
    const auto xmax = static_cast< double >( py::extract< float >( boxes[i].attr( "xmax" ) ) );
    const auto ymax = static_cast< double >( py::extract< float >( boxes[i].attr( "ymax" ) ) );

    const auto cls   = static_cast< unsigned int >( py::extract< long >( boxes[i].attr( "cls" ) ) );
    const auto score = static_cast< double >( py::extract< float >( boxes[i].attr( "score" ) ) );

    holes.emplace_back( xmin, xmax, ymin, ymax, cls, score );
  }

  return holes;
}

} // namespace

namespace detector
{

struct DnnHoleLocalizer::LocalizerImpl
{
  py::object localizer;
};

DnnHoleLocalizer::DnnHoleLocalizer( const std::string &model_path, int net_size, double score_threshold )
  : m_net_size{ net_size }
  , m_score_threshold{ score_threshold }
  , m_localizer_impl{ std::make_shared< LocalizerImpl >() }
{
  // Add python inria module path to PYTHONPATH env
  std::string python_path = getenv( "PYTHONPATH" );
  python_path += ":/tmp/python";
  setenv( "PYTHONPATH", python_path.c_str(), 1 );

  // Init python module
  init( model_path );
};

/*!
 * Initialize python env and load the TF model.
 */
void
DnnHoleLocalizer::init( const std::string &model_path )
{
  try
  {
    Py_Initialize();
    np::initialize();
  }
  catch ( const std::exception &e )
  {
    throw( std::runtime_error( "Cannot initialize python." ) );
  }

  try
  {
    const auto predict_module   = py::import( "python_deeplearning.predict" );
    const auto inference_class  = predict_module.attr( "Inference" );
    m_localizer_impl->localizer = inference_class( model_path );
  }
  catch ( const py::error_already_set & )
  {
    pythonException();
  }
}

/*!
 * Run the prediction (ie, hole detection and localisation) on an image.
 *
 * \param[in] img : Input image.
 * \return Vector of detected holes (bbox, score and class).
 */
std::optional< std::vector< data::DetectedHole2D > >
DnnHoleLocalizer::predict( const cv::Mat &img ) const
{
  try
  {
    // Resize input img
    cv::Mat resized_img;
    const cv::Size network_size{ m_net_size, m_net_size };
    cv::resize( img, resized_img, network_size, 0, 0, cv::INTER_LINEAR );

    // Convert input img to nd array
    const auto shape     = py::make_tuple( resized_img.rows, resized_img.cols, resized_img.channels() );
    const auto stride    = py::make_tuple( resized_img.channels() * resized_img.cols, resized_img.channels(), 1 );
    const auto data_type = np::dtype::get_builtin< uint8_t >();
    np::ndarray nd_image = np::from_data( resized_img.data, data_type, shape, stride, py::object() );

    // Predict
    auto predict = m_localizer_impl->localizer.attr( "predict" );
    return boxes_to_vector(
        py::extract< py::list >( predict( nd_image, img.rows, img.cols, m_net_size, m_score_threshold ) ) );
  }
  catch ( const py::error_already_set & )
  {
    pythonException();
    return std::nullopt;
  }
}

/*!
 * Hole detection and localisation.
 *
 * \param[in] img : Input image.
 * \return Vector of holes (bbox).
 */
std::vector< data::DetectedHole2D >
DnnHoleLocalizer::detect( const cv::Mat &img ) const
{
  return predict( img ).value_or( std::vector< data::DetectedHole2D >{} );
}

} // namespace detector
