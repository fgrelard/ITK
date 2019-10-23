/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "itkConfigure.h"


#ifndef ITK_USE_FFTWD
#  error "This program needs single precision FFTWD to work."
#endif


#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkCurvatureRegistrationFilter.h"
#include "itkFastSymmetricForcesDemonsRegistrationFunction.h"
#include "itkHistogramMatchingImageFilter.h"
#include "itkCastImageFilter.h"

#include "itkLinearInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"

#include "itkDisplacementFieldTransform.h"
#include "itkResampleImageFilter.h"

constexpr unsigned int Dimension = 2;

//  The following section of code implements a Command observer
//  that will monitor the evolution of the registration process.
//
class CommandIterationUpdate : public itk::Command
{
public:
  using Self = CommandIterationUpdate;
  using Superclass = itk::Command;
  using Pointer = itk::SmartPointer<CommandIterationUpdate>;
  itkNewMacro(CommandIterationUpdate);

protected:
  CommandIterationUpdate(){};

  using InternalImageType = itk::Image<float, Dimension>;
  using VectorPixelType = itk::Vector<float, Dimension>;
  using DisplacementFieldType = itk::Image<VectorPixelType, Dimension>;

  using RegistrationFilterType = itk::CurvatureRegistrationFilter<
    InternalImageType,
    InternalImageType,
    DisplacementFieldType,
    itk::FastSymmetricForcesDemonsRegistrationFunction<InternalImageType,
                                                       InternalImageType,
                                                       DisplacementFieldType>>;

public:
  void
  Execute(itk::Object * caller, const itk::EventObject & event) override
  {
    Execute((const itk::Object *)caller, event);
  }

  void
  Execute(const itk::Object * object, const itk::EventObject & event) override
  {
    const auto * filter = static_cast<const RegistrationFilterType *>(object);
    if (!(itk::IterationEvent().CheckEvent(&event)))
    {
      return;
    }
    std::cout << filter->GetMetric() << std::endl;
  }
};


template <typename DisplacementFieldType3D, typename DisplacementFieldType>
typename DisplacementFieldType3D::Pointer displacementField2DTo3D(const DisplacementFieldType& field) {
    using ImageIterator = itk::ImageRegionIteratorWithIndex<typename DisplacementFieldType::ObjectType>;
    ImageIterator  it( field, field->GetRequestedRegion() );

    typename DisplacementFieldType3D::Pointer field3D = DisplacementFieldType3D::New();
    typename DisplacementFieldType3D::RegionType region;
    typename DisplacementFieldType3D::IndexType  start;
    start[0] = 0;
    start[1] = 0;
    start[2] = 0;

    typename DisplacementFieldType3D::SizeType size;
    size[0] = field->GetLargestPossibleRegion().GetSize(0);
    size[1] = field->GetLargestPossibleRegion().GetSize(1);
    size[2] = 1;


    region.SetSize(size);
    region.SetIndex(start);

    typename DisplacementFieldType3D::SpacingType spacing;
    spacing[0] = field->GetSpacing()[0];
    spacing[1] = field->GetSpacing()[1];
    spacing[2] = 1;

    field3D->SetSpacing(spacing);
    field3D->SetRegions(region);
    field3D->Allocate();

    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
        typename DisplacementFieldType3D::IndexType index3D;
        index3D[0] = it.GetIndex()[0];
        index3D[1] = it.GetIndex()[1];
        index3D[2] = 0;

        typename DisplacementFieldType3D::PixelType v;
        for (int i = 0; i < 2; i++) {
            v.SetNthComponent(i, it.Get().GetElement(i));
        }
        v.SetNthComponent(2, 0);

        field3D->SetPixel(index3D, v);
    }
    return field3D;
}


int
main(int argc, char * argv[])
{
  if (argc < 4)
  {
    std::cerr << "Missing Parameters " << std::endl;
    std::cerr << "Usage: " << argv[0];
    std::cerr << " fixedImageFile movingImageFile ";
    std::cerr << " outputImageFile " << std::endl;
    return EXIT_FAILURE;
  }

  using PixelType = unsigned char;

  using FixedImageType = itk::Image<PixelType, Dimension>;
  using MovingImageType = itk::Image<PixelType, Dimension>;

  using FixedImageReaderType = itk::ImageFileReader<FixedImageType>;
  using MovingImageReaderType = itk::ImageFileReader<MovingImageType>;

  FixedImageReaderType::Pointer  fixedImageReader = FixedImageReaderType::New();
  MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();

  fixedImageReader->SetFileName(argv[1]);
  movingImageReader->SetFileName(argv[2]);

  using InternalPixelType = float;
  using InternalImageType = itk::Image<InternalPixelType, Dimension>;
  using FixedImageCasterType = itk::CastImageFilter<FixedImageType, InternalImageType>;
  using MovingImageCasterType =
    itk::CastImageFilter<MovingImageType, InternalImageType>;

  FixedImageCasterType::Pointer  fixedImageCaster = FixedImageCasterType::New();
  MovingImageCasterType::Pointer movingImageCaster = MovingImageCasterType::New();

  fixedImageCaster->SetInput(fixedImageReader->GetOutput());
  movingImageCaster->SetInput(movingImageReader->GetOutput());

  using MatchingFilterType =
    itk::HistogramMatchingImageFilter<InternalImageType, InternalImageType>;
  MatchingFilterType::Pointer matcher = MatchingFilterType::New();

  matcher->SetInput(movingImageCaster->GetOutput());
  matcher->SetReferenceImage(fixedImageCaster->GetOutput());
  matcher->SetNumberOfHistogramLevels(1024);
  matcher->SetNumberOfMatchPoints(7);
  matcher->ThresholdAtMeanIntensityOn();

  using VectorPixelType = itk::Vector<float, Dimension>;
  using VectorPixelType3D = itk::Vector<float, 3>;
  using DisplacementFieldType = itk::Image<VectorPixelType, Dimension>;
  using DisplacementFieldType3D = itk::Image<VectorPixelType3D, 3>;

  using RegistrationFilterType = itk::CurvatureRegistrationFilter<
    InternalImageType,
    InternalImageType,
    DisplacementFieldType,
    itk::FastSymmetricForcesDemonsRegistrationFunction<InternalImageType,
                                                       InternalImageType,
                                                       DisplacementFieldType> >;
  RegistrationFilterType::Pointer filter = RegistrationFilterType::New();

  CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  filter->AddObserver(itk::IterationEvent(), observer);

  filter->SetFixedImage(fixedImageCaster->GetOutput());
  filter->SetMovingImage(movingImageCaster->GetOutput());
  filter->SetNumberOfIterations(5);
  filter->SetTimeStep(1);
  filter->SetConstraintWeight(1);
  filter->Update();

  using InterpolatorPrecisionType = double;
  using TransformPrecisionType = float;
  using WarperType = itk::ResampleImageFilter<FixedImageType,
                                              MovingImageType,
                                              InterpolatorPrecisionType,
                                              TransformPrecisionType>;
  using InterpolatorType =
    itk::BSplineInterpolateImageFunction<MovingImageType, InterpolatorPrecisionType>;
  WarperType::Pointer       warper = WarperType::New();
  InterpolatorType::Pointer interpolator = InterpolatorType::New();
  FixedImageType::Pointer   fixedImage = fixedImageReader->GetOutput();

  warper->SetInput(movingImageReader->GetOutput());
  warper->SetInterpolator(interpolator);
  warper->UseReferenceImageOn();
  warper->SetReferenceImage(movingImageReader->GetOutput());

  using DisplacementFieldTransformType =
    itk::DisplacementFieldTransform<TransformPrecisionType, Dimension>;
  auto displacementTransform = DisplacementFieldTransformType::New();
  displacementTransform->SetDisplacementField(filter->GetOutput());
  warper->SetTransform(displacementTransform);

  DisplacementFieldType::Pointer field = filter->GetOutput();
  DisplacementFieldType3D::Pointer field3D = displacementField2DTo3D<DisplacementFieldType3D>(field);

  // Write warped image out to file
  using OutputPixelType = unsigned char;
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
  using CastFilterType = itk::CastImageFilter<MovingImageType, OutputImageType>;
  using WriterType = itk::ImageFileWriter<OutputImageType>;

  WriterType::Pointer     writer = WriterType::New();
  CastFilterType::Pointer caster = CastFilterType::New();

  writer->SetFileName(argv[3]);

  caster->SetInput(warper->GetOutput());
  writer->SetInput(caster->GetOutput());
  writer->Update();

  if (argc > 4) // if a fourth line argument has been provided...
  {

    using FieldWriterType = itk::ImageFileWriter<DisplacementFieldType3D>;

    FieldWriterType::Pointer fieldWriter = FieldWriterType::New();
    fieldWriter->SetFileName(argv[4]);
    fieldWriter->SetInput(field3D);

    try
    {
      fieldWriter->Update();
    }
    catch (itk::ExceptionObject & e)
    {
      e.Print(std::cerr);
    }
  }

  return EXIT_SUCCESS;
}
