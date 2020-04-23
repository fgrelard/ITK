#ifndef itkRegionalMaximaImageFilter_hxx
#define itkRegionalMaximaImageFilter_hxx

#include "itkValuedRadiusRegionalExtremaImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkNumericTraits.h"
#include "itkValuedRegionalExtremaImageFilter.h"
#include "itkProgressReporter.h"
#include "itkConnectedComponentAlgorithm.h"

namespace itk {

    template <typename TInputImage, typename TOutputImage, typename TFunction1, typename TFunction2>
    ValuedRadiusRegionalExtremaImageFilter<TInputImage, TOutputImage, TFunction1, TFunction2>::ValuedRadiusRegionalExtremaImageFilter()
        : Superclass(), m_Radius(1)

    {
        this->SetMarkerValue(NumericTraits<typename TOutputImage::PixelType>::max());
    }

    template <typename TInputImage, typename TOutputImage, typename TFunction1, typename TFunction2>
    void
    ValuedRadiusRegionalExtremaImageFilter<TInputImage, TOutputImage, TFunction1, TFunction2>::GenerateData()
    {
        // Allocate the output
        this->AllocateOutputs();

        const InputImageType * input = this->GetInput();
        OutputImageType *      output = this->GetOutput();

        // 2 phases
        ProgressReporter progress(this, 0, this->GetOutput()->GetRequestedRegion().GetNumberOfPixels() * 2);

        // copy input to output - isn't there a better way?
        using InputIterator = ImageRegionConstIterator<TInputImage>;
        using OutputIterator = ImageRegionIterator<TOutputImage>;

        InputIterator  inIt(input, output->GetRequestedRegion());
        OutputIterator outIt(output, output->GetRequestedRegion());
        inIt.GoToBegin();
        outIt.GoToBegin();

        InputImagePixelType firstValue = inIt.Get();
        bool flat = true;

        while (!outIt.IsAtEnd())
        {
            InputImagePixelType currentValue = inIt.Get();
            outIt.Set(static_cast<OutputImagePixelType>(currentValue));
            if (currentValue != firstValue)
            {
                flat = false;
            }
            ++inIt;
            ++outIt;
            progress.CompletedPixel();
        }
        // if the image is flat, there is no need to do the work:
        // the image will be unchanged
        if (!flat)
        {
            // Now for the real work!
            // More iterators - use shaped ones so that we can set connectivity
            // Note : all comments refer to finding regional minima, because
            // it is briefer and clearer than trying to describe both regional
            // maxima and minima processes at the same time
            ISizeType kernelRadius;
            kernelRadius.Fill(m_Radius);
            std::cout << kernelRadius << std::endl;

            NOutputIterator outNIt(kernelRadius, output, output->GetRequestedRegion());
            // setConnectivity(&outNIt, this->GetFullyConnected());

            ConstInputIterator inNIt(kernelRadius, input, output->GetRequestedRegion());
            // setConnectivity(&inNIt, this->GetFullyConnected());

            ConstantBoundaryCondition<OutputImageType> iBC;
            iBC.SetConstant(this->GetMarkerValue());
            inNIt.OverrideBoundaryCondition(&iBC);
            ConstantBoundaryCondition<OutputImageType> oBC;
            oBC.SetConstant(this->GetMarkerValue());
            outNIt.OverrideBoundaryCondition(&oBC);

            TFunction1 compareIn;
            TFunction2 compareOut;

            outIt.GoToBegin();
            // set up the stack and neighbor list
            IndexStack                              IS;

            while (!outIt.IsAtEnd())
            {
                OutputImagePixelType V = outIt.Get();
                // if the output pixel value = the marker value then we have
                // already visited this pixel and don't need to do so again
                if (compareOut(V, this->GetMarkerValue()))
                {
                    // reposition the input iterator
                    inNIt += outIt.GetIndex() - inNIt.GetIndex();

                    auto Cent = static_cast<InputImagePixelType>(V);

                    // check each neighbor of the input pixel
                    for (unsigned int i = 0; i < inNIt.Size(); ++i) {
                        InputImagePixelType Adjacent = inNIt.GetPixel(i);
                        if (compareIn(Adjacent, Cent)) {
                            // The centre pixel cannot be part of a regional minima
                            // because one of its neighbors is smaller.
                            // Set all pixels in the output image that are connected to
                            // the centre pixel and have the same value to
                            // m_MarkerValue
                            // This is the flood filling step. It is a simple, stack
                            // based, procedure. The original value (V) of the pixel is
                            // recorded and the pixel index in the output image
                            // is set to the marker value. The stack is initialized
                            // with the pixel index. The flooding procedure pops the
                            // stack, sets that index to the marker value and places the
                            // indexes of all neighbors with value V on the stack. The
                            // process terminates when the stack is empty.

                            outNIt += outIt.GetIndex() - outNIt.GetIndex();

                            OutputImagePixelType NVal;
                            OutIndexType         idx;
                            // Initialize the stack
                            outNIt.SetCenterPixel(this->GetMarkerValue());

                            while (!IS.empty())
                            {
                                // Pop the stack
                                idx = IS.top();
                                IS.pop();
                                // position the iterator
                                outNIt += idx - outNIt.GetIndex();
                                // check neighbors
                                for (unsigned int j = 0; j < outNIt.Size(); ++j)
                                {
                                    NVal = outNIt.GetPixel(j);
                                    if (NVal == V)
                                    {
                                        // still in a flat zone
                                        IS.push(outNIt.GetIndex(j));

                                        // set the output as the marker value
                                        outNIt.SetPixel(j, this->GetMarkerValue());
                                    }
                                }
                            }
                            // end flooding
                            break;
                        }
                    }
                }
                ++outIt;
                progress.CompletedPixel();
            }
        }
    }
}

#endif
