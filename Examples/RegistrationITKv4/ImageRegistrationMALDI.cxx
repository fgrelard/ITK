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

// Software Guide : BeginLatex
//
//  This example illustrates how to do registration with a 2D Rigid Transform
//  and with MutualInformation metric.
//
// Software Guide : EndLatex

#include "itkImageRegistrationMethod.h"

#include "itkImageRegistrationMethodv4.h"

#include "itkSimilarity2DTransform.h"
#include "itkCenteredTransformInitializer.h"
#include "itkBSplineTransform.h"

// Software Guide : BeginCodeSnippet
#include "itkMattesMutualInformationImageToImageMetricv4.h"

#include "itkNormalVariateGenerator.h"
#include "itkOnePlusOneEvolutionaryOptimizer.h"
#include "itkOnePlusOneEvolutionaryOptimizerv4.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"


#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkGradientDifferenceImageToImageMetric.h"


#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"


//  The following section of code implements a Command observer
//  used to monitor the evolution of the registration process.
//
#include "itkCommand.h"

template <typename OptimizerType>
class CommandIterationUpdate : public itk::Command
{
public:
    using Self = CommandIterationUpdate;
    using Superclass = itk::Command;
    using Pointer = itk::SmartPointer<Self>;
    itkNewMacro(Self);

protected:
    CommandIterationUpdate() = default;

public:

    using OptimizerPointer = const OptimizerType *;

    void
    Execute(itk::Object * caller, const itk::EventObject & event) override
    {
        Execute((const itk::Object *)caller, event);
    }

    void
    Execute(const itk::Object * object, const itk::EventObject & event) override
    {
        auto optimizer = static_cast<OptimizerPointer>(object);
        if (!itk::IterationEvent().CheckEvent(&event))
        {
            return;
        }
        std::cout << optimizer->GetCurrentIteration() << "   ";
        std::cout << optimizer->GetValue() << "   ";
        std::cout << optimizer->GetCurrentPosition() << std::endl;
    }
};


int
main(int argc, char * argv[])
{
    if (argc < 3)
    {
        std::cerr << "Missing Parameters " << std::endl;
        std::cerr << "Usage: " << argv[0];
        std::cerr << " fixedImageFile  movingImageFile ";
        std::cerr << "outputImagefile " << std::endl;
        return EXIT_FAILURE;
    }

    constexpr unsigned int Dimension = 2;
    using PixelType = float;

    using FixedImageType = itk::Image<PixelType, Dimension>;
    using MovingImageType = itk::Image<PixelType, Dimension>;

    // Software Guide : BeginLatex
    //
    // The Euler2DTransform applies a rigid transform in 2D space.
    //
    // Software Guide : EndLatex

    // Software Guide : BeginCodeSnippet
    //using TransformType = itk::Similarity2DTransform<double>;
    const unsigned int     SpaceDimension = Dimension;
    constexpr unsigned int SplineOrder = 3;
    using CoordinateRepType = double;

    using TransformType =
    itk::BSplineTransform<CoordinateRepType, SpaceDimension, SplineOrder>;

    using OptimizerTypeV4 = itk::OnePlusOneEvolutionaryOptimizerv4<double>;
    using OptimizerType = itk::OnePlusOneEvolutionaryOptimizer;

    using RegistrationTypeV4 =
        itk::ImageRegistrationMethodv4<FixedImageType, MovingImageType, TransformType>;
    using RegistrationType =
        itk::ImageRegistrationMethod<FixedImageType, MovingImageType>;

    using InterpolatorType = itk::LinearInterpolateImageFunction<MovingImageType, double>;


    // Software Guide : BeginCodeSnippet
    using MattesMetricTypeV4 =
        itk::MattesMutualInformationImageToImageMetricv4<FixedImageType, MovingImageType>;

    using GradientMetricType =
        itk::GradientDifferenceImageToImageMetric<FixedImageType, MovingImageType>;
    // Software Guide : EndCodeSnippet




    using FixedImageReaderType = itk::ImageFileReader<FixedImageType>;
    using MovingImageReaderType = itk::ImageFileReader<MovingImageType>;

    FixedImageReaderType::Pointer  fixedImageReader = FixedImageReaderType::New();
    MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();

    fixedImageReader->SetFileName(argv[1]);
    movingImageReader->SetFileName(argv[2]);
    FixedImageType::Pointer fixedImage = fixedImageReader->GetOutput();


    TransformType::Pointer    transform = TransformType::New();
    MattesMetricTypeV4::Pointer       metric = MattesMetricTypeV4::New();
    InterpolatorType::Pointer interpolator = InterpolatorType::New();

    OptimizerTypeV4::Pointer    optimizer = OptimizerTypeV4::New();
    RegistrationTypeV4::Pointer registration = RegistrationTypeV4::New();

    registration->SetFixedImage(fixedImageReader->GetOutput());
    registration->SetMovingImage(movingImageReader->GetOutput());

    fixedImageReader->Update();



    TransformType::PhysicalDimensionsType fixedPhysicalDimensions;
    TransformType::MeshSizeType           meshSize;
    TransformType::OriginType             fixedOrigin;

    for (unsigned int i = 0; i < SpaceDimension; i++)
    {
        fixedOrigin[i] = fixedImage->GetOrigin()[i];
        fixedPhysicalDimensions[i] =
            fixedImage->GetSpacing()[i] *
            static_cast<double>(fixedImage->GetLargestPossibleRegion().GetSize()[i] - 1);
    }
    unsigned int numberOfGridNodesInOneDimension = 7;
    meshSize.Fill(numberOfGridNodesInOneDimension - SplineOrder);

    transform->SetTransformDomainOrigin(fixedOrigin);
    transform->SetTransformDomainPhysicalDimensions(fixedPhysicalDimensions);
    transform->SetTransformDomainMeshSize(meshSize);
    transform->SetTransformDomainDirection(fixedImage->GetDirection());

    registration->SetOptimizer(optimizer);

    // registration->SetInterpolator(interpolator);

    // metric->SetDerivativeDelta(1.0);
    if (argc > 4) {
        int k = std::stoi(argv[4]);
        metric->SetNumberOfHistogramBins(k);
    } else {
        metric->SetNumberOfHistogramBins(10);
    }
    registration->SetMetric(metric);

    double samplingPercentage = 0.10;
    constexpr unsigned int numberOfLevels = 1;
    RegistrationTypeV4::MetricSamplingStrategyType samplingStrategy =
        RegistrationTypeV4::RANDOM;

    registration->SetNumberOfLevels(numberOfLevels);
    registration->SetMetricSamplingPercentage(samplingPercentage);
    registration->SetMetricSamplingStrategy(samplingStrategy);


    // using TransformInitializerType =
    //     itk::CenteredTransformInitializer<TransformType, FixedImageType, MovingImageType>;
    // TransformInitializerType::Pointer initializer = TransformInitializerType::New();
    // initializer->SetTransform(transform);
    // initializer->SetFixedImage(fixedImageReader->GetOutput());
    // initializer->SetMovingImage(movingImageReader->GetOutput());
    // initializer->GeometryOn();
    // initializer->InitializeTransform();

    // transform->SetScale(1.0);
    // transform->SetAngle(0.0);

    registration->SetInitialTransform(transform);
    registration->InPlaceOn();

    using GeneratorType = itk::Statistics::NormalVariateGenerator;

    GeneratorType::Pointer generator = GeneratorType::New();

    generator->Initialize(12345);

    optimizer->SetNormalVariateGenerator(generator);
    optimizer->Initialize(10);
    optimizer->SetInitialRadius(1.01);
    optimizer->SetGrowthFactor(-1.0);
    optimizer->SetShrinkFactor(-1.0);
    optimizer->SetEpsilon(0.00015);
    optimizer->SetMaximumIteration(10000);

    RegistrationTypeV4::ShrinkFactorsArrayType shrinkFactorsPerLevel;
    // shrinkFactorsPerLevel.SetSize(1);
    // shrinkFactorsPerLevel[0] = 1;

    // RegistrationTypeV4::SmoothingSigmasArrayType smoothingSigmasPerLevel;
    // smoothingSigmasPerLevel.SetSize(1);
    // smoothingSigmasPerLevel[0] = 0;

    // registration->SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel);
    // registration->SetShrinkFactorsPerLevel(shrinkFactorsPerLevel);

    // Create the Command observer and register it with the optimizer.
    //
    CommandIterationUpdate<OptimizerTypeV4>::Pointer observer = CommandIterationUpdate<OptimizerTypeV4>::New();
    optimizer->AddObserver(itk::IterationEvent(), observer);


    try
    {
        registration->Update();
        std::cout << "Optimizer stop condition = "
                  << registration->GetOptimizer()->GetStopConditionDescription()
                  << std::endl;
    }
    catch (itk::ExceptionObject & err)
    {
        std::cout << "ExceptionObject caught !" << std::endl;
        std::cout << err << std::endl;
        return EXIT_FAILURE;
    }

    using ParametersType = TransformType::ParametersType;
    ParametersType finalParameters = transform->GetParameters();

    const double finalAngle = finalParameters[0];
    const double finalTranslationX = finalParameters[1];
    const double finalTranslationY = finalParameters[2];

    const double rotationCenterX =
        registration->GetOutput()->Get()->GetFixedParameters()[0];
    const double rotationCenterY =
        registration->GetOutput()->Get()->GetFixedParameters()[1];

    unsigned int numberOfIterations = optimizer->GetCurrentIteration();

    double bestValue = optimizer->GetValue();

    // Print out results
    //

    const double finalAngleInDegrees = finalAngle * 180 / itk::Math::pi;

    std::cout << "Result = " << std::endl;
    std::cout << " Angle (radians) " << finalAngle << std::endl;
    std::cout << " Angle (degrees) " << finalAngleInDegrees << std::endl;
    std::cout << " Translation X  = " << finalTranslationX << std::endl;
    std::cout << " Translation Y  = " << finalTranslationY << std::endl;
    std::cout << " Fixed Center X = " << rotationCenterX << std::endl;
    std::cout << " Fixed Center Y = " << rotationCenterY << std::endl;
    std::cout << " Iterations     = " << numberOfIterations << std::endl;
    std::cout << " Metric value   = " << bestValue << std::endl;


    using ResampleFilterType = itk::ResampleImageFilter<MovingImageType, FixedImageType>;

    ResampleFilterType::Pointer resample = ResampleFilterType::New();

    resample->SetTransform(transform);
    resample->SetInput(movingImageReader->GetOutput());


    resample->SetSize(fixedImage->GetLargestPossibleRegion().GetSize());
    resample->SetOutputOrigin(fixedImage->GetOrigin());
    resample->SetOutputSpacing(fixedImage->GetSpacing());
    resample->SetOutputDirection(fixedImage->GetDirection());
    resample->SetDefaultPixelValue(0);

    using OutputPixelType = unsigned char;

    using OutputImageType = itk::Image<OutputPixelType, Dimension>;

    using CastFilterType = itk::CastImageFilter<FixedImageType, OutputImageType>;

    using WriterType = itk::ImageFileWriter<OutputImageType>;

    WriterType::Pointer     writer = WriterType::New();
    CastFilterType::Pointer caster = CastFilterType::New();

    writer->SetFileName(argv[3]);

    caster->SetInput(resample->GetOutput());
    writer->SetInput(caster->GetOutput());
    writer->Update();

    return EXIT_SUCCESS;
}

//  Software Guide : BeginLatex
//
//  Let's execute this example over some of the images provided in
//  \code{Examples/Data}, for example:
//
//  \begin{itemize}
//  \item \code{BrainProtonDensitySlice.png}
//  \item \code{BrainProtonDensitySliceR10X13Y17.png}
//  \end{itemize}
//
//  The second image is the result of intentionally rotating the first
//  image by $10$ degrees and shifting it $13mm$ in $X$ and $17mm$ in
//  $Y$. Both images have unit-spacing and are shown in Figure
//  \ref{fig:FixedMovingImageRegistration5}. The example
//  yielded the following results.
//
//  \begin{verbatim}
//
//  Angle (radians) 0.174569
//  Angle (degrees) 10.0021
//  Translation X = 13.0958
//  Translation Y = 15.9156
//
//  \end{verbatim}
//
//  These values match the true misalignment introduced in the moving image.
//
//  Software Guide : EndLatex
