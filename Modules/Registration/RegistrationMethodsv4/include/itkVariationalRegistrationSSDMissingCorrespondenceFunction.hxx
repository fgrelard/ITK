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
#ifndef itkVariationalRegistrationSSDMissingCorrespondenceFunction_hxx
#define itkVariationalRegistrationSSDMissingCorrespondenceFunction_hxx

#include "itkVariationalRegistrationSSDMissingCorrespondenceFunction.h"
#include "itkMath.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkDanielssonDistanceMapImageFilter.h"
#include "itkMinimumMaximumImageFilter.h"
#include "itkImageDuplicator.h"

#include "itkVectorMagnitudeVarianceImageFilter.h"
#include "itkVectorDivergenceImageFilter.h"
#include "itkVectorMagnitudeImageFilter.h"
#include "itkDisplacementFieldJacobianDeterminantFilter.h"

#include "itkGaussianOperator.h"
#include "itkNeighborhoodInnerProduct.h"
#include "itkAbsImageFilter.h"

#include <algorithm>

namespace itk
{

/**
 * Default constructor
 */
  template< typename TFixedImage, typename TMovingImage, typename TDisplacementField >
  VariationalRegistrationSSDMissingCorrespondenceFunction< TFixedImage, TMovingImage, TDisplacementField >
  ::VariationalRegistrationSSDMissingCorrespondenceFunction()
  {
    RadiusType r;
    for( unsigned int j = 0; j < ImageDimension; j++ )
    {
      r[j] = 0;
    }
    this->SetRadius( r );

    m_IntensityDifferenceThreshold = 0.001;

    m_Normalizer = 1.0;
    m_Radius = 7.0;
    m_FixedImageGradientCalculator = GradientCalculatorType::New();
    m_WarpedImageGradientCalculator = GradientCalculatorType::New();
    m_WeightImageGradientCalculator = GradientCalculatorType::New();

    m_GradientType = GRADIENT_TYPE_WARPED;

    m_WeightImage = nullptr;
    m_DivergenceImage = nullptr;
  }


  template <typename TFixedImage, typename TMovingImage, typename TDisplacementField>
  void
  VariationalRegistrationSSDMissingCorrespondenceFunction<TFixedImage, TMovingImage, TDisplacementField>::ComputeDivergence() {

    using DivergenceFilter =
      itk::VectorDivergenceImageFilter<DisplacementFieldType> ;
    using VarianceFilter =
      itk::VectorMagnitudeVarianceImageFilter<DisplacementFieldType>;
    using MagnitudeFilter =
      itk::VectorMagnitudeImageFilter<DisplacementFieldType, DivergenceImage>;
    using JacobianFilter =
      itk::DisplacementFieldJacobianDeterminantFilter<DisplacementFieldType, RealType, DivergenceImage>;

    using ImageIterator = itk::ImageRegionIteratorWithIndex<DivergenceImage>;
    using IteratorType = itk::ImageRegionIterator<DivergenceImage>;
    using DuplicatorType = itk::ImageDuplicator<DivergenceImage>;

    DisplacementFieldTypePointer field = this->GetDisplacementField();
    typename DivergenceFilter::Pointer divergenceFilter = DivergenceFilter::New();
    divergenceFilter->SetUsePrincipleComponentsOff();
    divergenceFilter->SetUseImageSpacingOn();
    divergenceFilter->SetInput(field);
    divergenceFilter->Update();

    typename MagnitudeFilter::Pointer magnitudeFilter = MagnitudeFilter::New();
    magnitudeFilter->SetInput(field);
    magnitudeFilter->Update();

    typename JacobianFilter::Pointer jacobianFilter = JacobianFilter::New();
    jacobianFilter->SetUseImageSpacingOn();
    jacobianFilter->SetInput(field);
    jacobianFilter->Update();

    typename VarianceFilter::Pointer varianceFilter = VarianceFilter::New();
    varianceFilter->SetUsePrincipleComponentsOff();
    varianceFilter->SetUseImageSpacingOn();
    varianceFilter->SetInput(field);
    varianceFilter->Update();
    DivergenceImagePointer gradientImage = divergenceFilter->GetOutput();
    DivergenceImagePointer varianceImage = varianceFilter->GetOutput();
    DivergenceImagePointer magnitudeImage = magnitudeFilter->GetOutput();
    DivergenceImagePointer jacobianImage = jacobianFilter->GetOutput();

    typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
    duplicator->SetInputImage(gradientImage);
    duplicator->Update();
    m_DivergenceImage = duplicator->GetOutput();

    ImageIterator  it( m_DivergenceImage, m_DivergenceImage->GetRequestedRegion() );
    it.GoToBegin();

    while( !it.IsAtEnd() )
    {

      typename DivergenceImage::IndexType index3D = it.GetIndex();
      typename DivergenceImage::PixelType d = it.Get();
      typename DivergenceImage::PixelType j = jacobianImage->GetPixel(index3D);
      typename DivergenceImage::PixelType v = varianceImage->GetPixel(index3D);
      typename DivergenceImage::PixelType m = magnitudeImage->GetPixel(index3D);
      it.Set( j );
      ++it;
    }
  }

  template  <typename TFixedImage, typename TMovingImage, typename TDisplacementField>
  typename VariationalRegistrationSSDMissingCorrespondenceFunction<TFixedImage, TMovingImage, TDisplacementField>::
  RealType
  VariationalRegistrationSSDMissingCorrespondenceFunction<TFixedImage, TMovingImage, TDisplacementField>::MinimumRadiusDivergenceChange(const NeighborhoodIteratorType& it) {
    typename NeighborhoodIteratorType::RadiusType radius;


    for (size_t r = 0; r < m_Radius; r++) {
      RealType currentRadius = r+1;
      for (unsigned int i = 0; i < DivergenceImage::ImageDimension; ++i) radius[i] = currentRadius;
      radius[2] = 0;

      NeighborhoodIteratorType itClone(radius, m_DivergenceImage, m_DivergenceImage->GetRequestedRegion());
      itClone.SetLocation(it.GetIndex());
      itClone.SetRadius(radius);

      bool pos = false, neg = false;
      for (unsigned int i = 0; i < itClone.Size(); ++i)
      {
        auto value = itClone.GetPixel(i);
        if (value < 0) neg = true;
        else if (value > 0) pos = true;
      }
      if (pos && neg) {
        return currentRadius;
      }
    }
    return m_Radius;
  }

   template <typename TFixedImage, typename TMovingImage, typename TDisplacementField>
  void
  VariationalRegistrationSSDMissingCorrespondenceFunction<TFixedImage, TMovingImage, TDisplacementField>::ComputeWeights() {
    using ImageIterator = itk::ImageRegionIterator<DivergenceImage>;
    using GaussianOperator = itk::GaussianOperator<RealType, ImageDimension>;
    using InnerProduct = itk::NeighborhoodInnerProduct<DivergenceImage>;
    using MinMaxFilter = itk::MinimumMaximumImageFilter<TMovingImage>;
    using AbsImageFilter = itk::AbsImageFilter<DivergenceImage, DivergenceImage>;


    typename DivergenceImage::Pointer tmpWeightImage = DivergenceImage::New();
    auto fixed = this->GetFixedImage();
    typename DivergenceImage::IndexType start;
    start[0] = 0; // first index on X
    start[1] = 0; // first index on Y
    start[2] = 0; // first index on Z

    typename DivergenceImage::SizeType size;
    auto sizeFixed = fixed->GetLargestPossibleRegion().GetSize();
    size[0] = sizeFixed[0]; // size along X
    size[1] = sizeFixed[1]; // size along Y
    size[2] = sizeFixed[2]; // size along Z

    typename DivergenceImage::RegionType region;
    region.SetSize(size);
    region.SetIndex(start);
    tmpWeightImage->SetRegions(region);
    tmpWeightImage->Allocate();
    tmpWeightImage->FillBuffer(0);

    typename AbsImageFilter::Pointer absFilter = AbsImageFilter::New();
    absFilter->SetInput(m_DivergenceImage);
    absFilter->Update();
    DivergenceImagePointer absDivImage = absFilter->GetOutput();


    typename NeighborhoodIteratorType::RadiusType radius;
    for (unsigned int i = 0; i < DivergenceImage::ImageDimension; ++i) radius[i] = m_Radius;
    radius[2] = 0;
    NeighborhoodIteratorType it( radius, m_DivergenceImage, m_DivergenceImage->GetRequestedRegion() );
    ImageIterator out( tmpWeightImage, tmpWeightImage->GetRequestedRegion() );


    //Gaussian operator
    RealType kernelSize = 2 * m_Radius + 1;
    RealType sigma = kernelSize / 4;
    GaussianOperator gaussianOperator;
    gaussianOperator.SetVariance(sigma*sigma);
    gaussianOperator.CreateToRadius(radius);

    InnerProduct innerProduct;
    it.GoToBegin();
    out.GoToBegin();
    while ( !it.IsAtEnd() ) {
      bool pos = false, neg = false;
      for (unsigned int i = 0; i < it.Size(); ++i)
      {
        auto value = it.GetPixel(i);
        if (value < 0) neg = true;
        else if (value > 0) pos = true;
      }
      if (pos && neg) {
        //Absolute values
        NeighborhoodIteratorType cloneIt(radius, absDivImage, absDivImage->GetRequestedRegion());
        cloneIt.SetLocation(it.GetIndex());
        RealType average = innerProduct(cloneIt, gaussianOperator) / (float)(cloneIt.Size());
        RealType minRadius = this->MinimumRadiusDivergenceChange(it);
        RealType radiusFunction = std::exp(-std::pow(minRadius-1, 2)/1.0);
        out.Set(average*radiusFunction);
      }
      // auto value = it.GetCenterPixel();
      // if (value < 0) {
      //   out.Set(value);
      // }
      ++it;
      ++out;
    }

    typename MinMaxFilter::Pointer minmax = MinMaxFilter::New();
    minmax->SetInput(tmpWeightImage);
    minmax->Update();

    RealType max_value = minmax->GetMaximum();
    RealType max_bound = max_value;
    ImageIterator inWeight( tmpWeightImage, tmpWeightImage->GetRequestedRegion() );
    ImageIterator outWeight( m_WeightImage, m_WeightImage->GetRequestedRegion() );
    inWeight.GoToBegin();
    outWeight.GoToBegin();
    while ( !outWeight.IsAtEnd() ) {
      RealType previousW = outWeight.Get();
      if (std::isnan(previousW)) {
        previousW = 1.0;
      }
      auto index = outWeight.GetIndex();
      if ((index[0] >= 39 && index[0] <= 44 && index[1] >= 39 && index[1] <= 44) ||
          ((index[0] >= 57 && index[0] <= 63 && index[1] >= 50 && index[1] <= 56))) {
        previousW = 0.0;
      }
      RealType w = inWeight.Get();
      RealType minValue = 0.0;
      RealType outW = previousW - (1.0 - std::exp(-(w*w)/(2*max_bound*max_bound)));
      outW = std::max(outW, minValue);
      outW = std::exp(-(w*w)/(2*max_bound*max_bound));
      outWeight.Set(outW);
      outWeight.Set(previousW);
      ++outWeight;
      ++inWeight;
    }
  }

  template <typename TFixedImage, typename TMovingImage, typename TDisplacementField>
  void
  VariationalRegistrationSSDMissingCorrespondenceFunction<TFixedImage, TMovingImage, TDisplacementField>::ComputeWeights2() {

    using ImageIterator = itk::ImageRegionIterator<DivergenceImage>;
    using InnerProduct = itk::NeighborhoodInnerProduct<DivergenceImage>;
    using MinMaxFilter = itk::MinimumMaximumImageFilter<TMovingImage>;


    typename MinMaxFilter::Pointer minmax = MinMaxFilter::New();
    minmax->SetInput(m_DivergenceImage);
    minmax->Update();

    RealType max_value = minmax->GetMaximum();
    RealType max_bound = max_value/2;
    ImageIterator in( m_DivergenceImage, m_DivergenceImage->GetRequestedRegion() );
    ImageIterator outWeight( m_WeightImage, m_WeightImage->GetRequestedRegion() );

    in.GoToBegin();
    outWeight.GoToBegin();
    while ( !outWeight.IsAtEnd() ) {
      RealType w = in.Get();
      RealType outW = 1.0 - std::exp(-(w*w)/(2*max_bound*max_bound));
      outWeight.Set(outW);
      ++outWeight;
      ++in;
    }
  }

  template <typename TFixedImage, typename TMovingImage, typename TDisplacementField>
  void
  VariationalRegistrationSSDMissingCorrespondenceFunction<TFixedImage, TMovingImage, TDisplacementField>::UpdateValues() {
    WarpedImagePointer warped = Superclass::GetWarpedImage();
    using BinaryFilterType = itk::BinaryThresholdImageFilter<TMovingImage, TMovingImage>;
    typename BinaryFilterType::Pointer binary = BinaryFilterType::New();
    binary->SetInput(warped);
    binary->SetLowerThreshold(1.0);
    binary->SetOutsideValue(255);
    binary->SetInsideValue(0);
    binary->Update();

    using DanielssonDistanceMapImageFilter =
      itk::DanielssonDistanceMapImageFilter<TMovingImage, TMovingImage>;
    typename DanielssonDistanceMapImageFilter::Pointer distanceMapImageFilter =
      DanielssonDistanceMapImageFilter::New();
    distanceMapImageFilter->SetInput(binary->GetOutput());
    distanceMapImageFilter->Update();
    typename TMovingImage::Pointer distanceMap = distanceMapImageFilter->GetOutput();

    using DuplicatorType = itk::ImageDuplicator<TMovingImage>;
    using MinMaxFilter = itk::MinimumMaximumImageFilter<TMovingImage>;

    typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
    duplicator->SetInputImage(distanceMap);
    duplicator->Update();
    typename TMovingImage::Pointer clonedImage = duplicator->GetOutput();

    typename MinMaxFilter::Pointer minmax = MinMaxFilter::New();
    minmax->SetInput(clonedImage);
    minmax->Update();

    auto max_value = minmax->GetMaximum();
    auto min_value = minmax->GetMinimum();

    minmax->SetInput(warped);
    minmax->Update();

    auto prevmax_value = minmax->GetMaximum();
    auto prevmin_value = minmax->GetMinimum();


    std::cout << "Previous=" << prevmin_value << " " << prevmax_value << " Next=" << min_value << " " << max_value << std::endl;
    using RescaleFilterType = itk::RescaleIntensityImageFilter<TMovingImage, TMovingImage>;
    typename RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
    rescaleFilter->SetInput(warped);
    rescaleFilter->SetOutputMinimum(min_value);
    rescaleFilter->SetOutputMaximum(max_value);
    rescaleFilter->Update();

    auto rescaled = rescaleFilter->GetOutput();
    this->SetWarpedImage(rescaled);
  }

/**
 * Set the function state values before each iteration
 */
  template< typename TFixedImage, typename TMovingImage, typename TDisplacementField >
  void
  VariationalRegistrationSSDMissingCorrespondenceFunction< TFixedImage, TMovingImage, TDisplacementField >
  ::InitializeIteration()
  {
    // Call superclass method
    Superclass::InitializeIteration();

    if ( !this->GetWeightImage() ) {
      m_WeightImage = DivergenceImage::New();

      auto fixed = this->GetFixedImage();
      typename DivergenceImage::IndexType start;
      start[0] = 0; // first index on X
      start[1] = 0; // first index on Y
      start[2] = 0; // first index on Z

      typename DivergenceImage::SizeType size;
      auto sizeFixed = fixed->GetLargestPossibleRegion().GetSize();
      size[0] = sizeFixed[0]; // size along X
      size[1] = sizeFixed[1]; // size along Y
      size[2] = sizeFixed[2]; // size along Z

      typename DivergenceImage::RegionType region;
      region.SetSize(size);
      region.SetIndex(start);
      m_WeightImage->SetRegions(region);
      m_WeightImage->Allocate();
      m_WeightImage->FillBuffer(1.0);
    }

    //Update DT values taking into account the deformed shape
    this->UpdateValues();
    this->ComputeDivergence();
    this->ComputeWeights2();

    // cache fixed image information
    SpacingType fixedImageSpacing = this->GetFixedImage()->GetSpacing();
    m_ZeroUpdateReturn.Fill( 0.0 );

    // compute the normalizer
    m_Normalizer = 0.0;
    for( unsigned int k = 0; k < ImageDimension; k++ )
    {
      m_Normalizer += fixedImageSpacing[k] * fixedImageSpacing[k];
    }
    m_Normalizer /= static_cast< double >( ImageDimension );

    // setup gradient calculator
    m_WarpedImageGradientCalculator->SetInputImage( this->GetWarpedImage() );
    m_FixedImageGradientCalculator->SetInputImage( this->GetFixedImage() );
    m_WeightImageGradientCalculator->SetInputImage ( this->GetWeightImage() );
  }

/**
 * Compute update at a specific neighborhood
 */
  template< typename TFixedImage, typename TMovingImage, typename TDisplacementField >
  typename VariationalRegistrationSSDMissingCorrespondenceFunction< TFixedImage, TMovingImage, TDisplacementField >
  ::PixelType
  VariationalRegistrationSSDMissingCorrespondenceFunction< TFixedImage, TMovingImage, TDisplacementField >
  ::ComputeUpdate( const NeighborhoodType &it, void * gd,
                   const FloatOffsetType& itkNotUsed(offset) )
  {
    // Get fixed image related information
    // Note: no need to check the index is within
    // fixed image buffer. This is done by the external filter.
    const IndexType index = it.GetIndex();

    // Check if index lies inside mask
    const MaskImageType * mask = this->GetMaskImage();
    if( mask && (mask->GetPixel( index ) <= this->GetMaskBackgroundThreshold()) )
    {
      return m_ZeroUpdateReturn;
    }

    const auto warpedValue = (double) this->GetWarpedImage()->GetPixel( index );
    const auto fixedValue = (double) this->GetFixedImage()->GetPixel( index );
    auto weight = this->GetWeightImage()->GetPixel( index );
    if (std::isnan(weight)) {
      if (index[0] >= 39 && index[0] <= 44 && index[1] >= 39 && index[1] <= 44) {
        weight = 0.0;
      }
      weight = 1.0;
    }

    // typename GradientCalculatorType::OutputType gradientWeight = m_WeightImageGradientCalculator->EvaluateAtIndex( index );
    // auto norm = gradientWeight.Normalize();
    // IndexType newPoint = index;
    // int i = 0;
    // while (norm > 0 && i < m_Radius) {
    //   itk::Offset<3> off;
    //   off[0] = gradientWeight[0];
    //   off[1] = gradientWeight[1];
    //   off[2] = gradientWeight[2];
    //   newPoint += off;
    //   gradientWeight = m_WeightImageGradientCalculator->EvaluateAtIndex( newPoint );
    //   norm = gradientWeight.Normalize();
    //   i++;
    // }
    // if (index != newPoint && index[0] >= 39 && index[0] <= 44 && index[1] >= 39 && index[1] <= 44) {
    //   std::cout << norm << std::endl;
    //   std::cout << index << newPoint << std::endl;
    // }


    // Calculate spped value
    const double speedValue = fixedValue - warpedValue;
    const double sqr_speedValue = itk::Math::sqr( speedValue );

    // Calculate update
    PixelType update;
    if( itk::Math::abs( speedValue ) < m_IntensityDifferenceThreshold )
    {
      update = m_ZeroUpdateReturn;
    }
    else
    {
      typename GradientCalculatorType::OutputType gradient;

      // Compute the gradient of either fixed or moving image
      if( m_GradientType == GRADIENT_TYPE_WARPED )
      {
        gradient = m_WarpedImageGradientCalculator->EvaluateAtIndex( index );
      }
      else
        if( m_GradientType == GRADIENT_TYPE_FIXED )
        {
          gradient = m_FixedImageGradientCalculator->EvaluateAtIndex( index );
        }
        else
          if( m_GradientType == GRADIENT_TYPE_SYMMETRIC )
          {
            // Does not have to be divided by 2, normalization is done afterwards
            gradient = m_WarpedImageGradientCalculator->EvaluateAtIndex( index )
              + m_FixedImageGradientCalculator->EvaluateAtIndex( index );
          }
          else
          {
            itkExceptionMacro( << "Unknown gradient type!" );
          }

      for( unsigned int j = 0; j < ImageDimension; j++ )
      {
        // if (weight == 1)
          update[j] = speedValue * gradient[j];
        // if (weight == 0)
        //   update[j] = 1.6;
      }
    }

    // Update the global data (metric etc.)
    auto *globalData = (GlobalDataStruct *) gd;
    if( globalData )
    {
      globalData->m_NumberOfPixelsProcessed += 1;
      globalData->m_SumOfMetricValues += sqr_speedValue;
      globalData->m_SumOfSquaredChange += update.GetSquaredNorm();
    }

    return update;
  }

/**
 * Sets the warped moving image.
 */
  template< typename TFixedImage, typename TMovingImage, typename TDisplacementField >
  void
  VariationalRegistrationSSDMissingCorrespondenceFunction< TFixedImage, TMovingImage, TDisplacementField >
  ::SetWarpedImage(const WarpedImagePointer& warped)
  {
    m_WarpedImageDT = warped;
  }


/**
 * Return the warped moving image.
 */
  template< typename TFixedImage, typename TMovingImage, typename TDisplacementField >
  const typename VariationalRegistrationSSDMissingCorrespondenceFunction< TFixedImage, TMovingImage, TDisplacementField >
  ::WarpedImagePointer
  VariationalRegistrationSSDMissingCorrespondenceFunction< TFixedImage, TMovingImage, TDisplacementField >
  ::GetWarpedImage() const
  {
    return m_WarpedImageDT;
  }


/**
 * Standard "PrintSelf" method.
 */
  template< typename TFixedImage, typename TMovingImage, typename TDisplacementField >
  void
  VariationalRegistrationSSDMissingCorrespondenceFunction< TFixedImage, TMovingImage, TDisplacementField >
  ::PrintSelf( std::ostream& os, Indent indent ) const
  {
    Superclass::PrintSelf( os, indent );

    os << indent << "FixedImageGradientCalculator: ";
    os << m_FixedImageGradientCalculator.GetPointer() << std::endl;
    os << indent << "WarpedImageGradientCalculator: ";
    os << m_WarpedImageGradientCalculator.GetPointer() << std::endl;
    os << indent << "GradientType: ";
    os << m_GradientType << std::endl;

    os << indent << "IntensityDifferenceThreshold: ";
    os << m_IntensityDifferenceThreshold << std::endl;
    os << indent << "Normalizer: ";
    os << m_Normalizer << std::endl;

  }

} // end namespace itk

#endif
