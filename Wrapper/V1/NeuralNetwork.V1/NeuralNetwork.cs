using System.Runtime.InteropServices;
using Base;

namespace NeuralNetwork.V1;

public partial class NeuralNetwork : INeuralNetwork
{
    private readonly IntPtr _ptr;
    
    public int Length { get; }

    public NeuralNetwork(params int[] layers)
    {
        _ptr = Create(layers.Length, layers);
        Length = GetCount(_ptr);
        InitializeWeightsAndBiases();
    }

    private NeuralNetwork(IntPtr ptr)
    {
        _ptr = ptr;
        Length = GetCount(_ptr);
    }

    public static NeuralNetwork Restore(string file)
    {
        var ptr = FromFile(file);
        if (ptr == IntPtr.Zero) throw new Exception("Restore failed");
        return new NeuralNetwork(ptr);
    }

    public void Save(string file)
    {
        if(Save(_ptr, file) != 0) throw new Exception("Save failed");
    }

    public IntPtr GetPointer() => _ptr;

    public void InitializeWeightsAndBiases()
    {
        Initialize(_ptr);
    }

    public int GetInCount(int layer)
    {
        if (layer < 0 || layer >= GetCount(_ptr)) throw new IndexOutOfRangeException();
        return GetInCount(_ptr, layer);
    }

    public int GetInputInCount()
    {
        return GetInCount(_ptr, 0);
    }

    public int GetOutCount(int layer)
    {
        if (layer < 0 || layer >= GetCount(_ptr)) throw new IndexOutOfRangeException();
        return GetOutCount(_ptr, layer);
    }

    public int GetOutputOutCount()
    {
        return GetOutCount(_ptr, Length - 1);
    }

    public void SetWeight(int layer, int input, int output, double value)
    {
        CheckBounds(layer, input, output);
        SetWeight(_ptr, layer, input, output, value);
    }

    public double GetWeight(int layer, int input, int output)
    {
        CheckBounds(layer, input, output);
        return GetWeight(_ptr, layer, input, output);
    }
    
    public void SetBias(int layer, int output, double value)
    {
        if (output < 0 || output >= GetOutCount(layer)) throw new IndexOutOfRangeException();
        SetBias(_ptr, layer, output, value);
    }
    
    public double GetBias(int layer, int output)
    {
        if (output < 0 || output >= GetOutCount(layer)) throw new IndexOutOfRangeException();
        return GetBias(_ptr, layer, output);
    }

    public void SetAllWeightsAndBiases(double weights, double biases)
    {
        SetAllWeightsAndBiases(_ptr, weights, biases);
    }

    public double GetLearningRate() => GetLearningRate(_ptr);
    public void SetLearningRate(double lr) => SetLearningRate(_ptr, lr);
    public int GetShuffleDataOnIteration() => GetShuffleDataOnIteration(_ptr);
    public void SetShuffleDataOnIteration(int sdoi) => SetShuffleDataOnIteration(_ptr, sdoi);
    public int GetThreadCount() => GetThreadCount(_ptr);
    public void SetThreadCount(int count) => SetThreadCount(_ptr, count);
    public void SetOptimizerGradientDescent() => SetOptimizerGradientDescent(_ptr);
    public void SetOptimizerMomentum(double momentum) => SetOptimizerMomentum(_ptr, momentum);
    public void SetOptimizerNesterov(double decay) => SetOptimizerNesterov(_ptr, decay);
    public void SetOptimizerAdam(double delta1, double delta2) => SetOptimizerAdam(_ptr, delta1, delta2);
    public void SetDataSelectorFullBatch() => SetDataSelectorFullBatch(_ptr);
    public void SetDataSelectorMiniBatch(int batchSize) => SetDataSelectorMiniBatch(_ptr, batchSize);
    public void SetSchedulerConstant() => SetSchedulerConstant(_ptr);
    public void SetSchedulerIterationDecay(double proportion) => SetSchedulerIterationDecay(_ptr, proportion);
    public void SetSchedulerExponentialDecay(double decay) => SetSchedulerExponentialDecay(_ptr, decay);
    public void SetSchedulerInverseDecay(double decay) => SetSchedulerInverseDecay(_ptr, decay);
    public void SetSchedulerCosineDecay(double endLearningRate, int iterationSpan) =>
        SetSchedulerCosineDecay(_ptr, endLearningRate, iterationSpan);

    public void SetActivationType(ActivationType type, ActivationType outputType)
    {
        SetActivationType(_ptr, (int)type, (int)outputType);
    }
    
    public void SetCostType(CostType type)
    {
        SetCostType(_ptr, (int)type);
    }

    public double[] Predict(double[] inputs) => Predict(inputs, inputs.Length);
    
    public double[] Predict(double[] inputs, int length)
    {
        if (length != GetInCount(0))
            throw new ArgumentException("Inputs length does not correspond to the first layer of the neural network");
        
        var count = GetOutCount(Length - 1);
        var arr = new double[count]; //TODO Look into using stackalloc when output is a single int
        
        Predict(_ptr, inputs, length, arr, count);
        return arr;
    }
    
    public void Learn(FlattenedData data, int iterations, ILearningState? state)
    {
        var count = data.GetCount();
        if (count == 0) return;
        
        var statePtr = state?.GetPointer() ?? IntPtr.Zero;
        Learn(_ptr, statePtr, data.Inputs, data.InputCutOff, data.Expected, data.ExpectedCutOff, count,
            iterations);
    }

    public double GetCost(double[] inputs, double[] expected)
    {
        CheckTestDataBounds(inputs.Length, expected.Length);
        return Cost(_ptr, inputs, inputs.Length, expected, expected.Length); 
    }

    public double GetCost(FlattenedData data)
    {
        var count = data.GetCount();
        if (count == 0) return 0;
        
        CheckTestDataBounds(data.InputCutOff, data.ExpectedCutOff);
        return MultiCost(_ptr, data.Inputs, data.InputCutOff, data.Expected, data.ExpectedCutOff, count);
    }

    public void Dispose()
    {
        GC.SuppressFinalize(this);
        Dispose(_ptr);
    }

    private void CheckBounds(int layer, int input, int output)
    {
        if (input < 0 || input >= GetInCount(layer) || output < 0 || output >= GetOutCount(layer))
            throw new IndexOutOfRangeException();
    }

    private void CheckTestDataBounds(int inputCount, int outputCount)
    {
        if (inputCount != GetInCount(0) || outputCount != GetOutCount(Length - 1))
            throw new ArgumentException("Test data input or output count incorrect");
    }

    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial IntPtr Create(int count, int[] layers);

    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void Dispose(IntPtr ptr);

    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial int GetCount(IntPtr ptr);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial int GetOutCount(IntPtr ptr, int layer);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial int GetInCount(IntPtr ptr, int layer);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetWeight(IntPtr ptr, int layer, int input, int output, double value);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial double GetWeight(IntPtr ptr, int layer, int input, int output);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetBias(IntPtr ptr, int layer, int output, double value);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial double GetBias(IntPtr ptr, int layer, int output);

    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetAllWeightsAndBiases(IntPtr ptr, double weights, double biases);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial double GetLearningRate(IntPtr ptr);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetLearningRate(IntPtr ptr, double lr);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial int GetShuffleDataOnIteration(IntPtr ptr);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetShuffleDataOnIteration(IntPtr ptr, int sdoi);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial int GetThreadCount(IntPtr ptr);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetThreadCount(IntPtr ptr, int th);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetActivationType(IntPtr ptr, int type, int outputType);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetCostType(IntPtr ptr, int type);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetOptimizerGradientDescent(IntPtr ptr);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetOptimizerMomentum(IntPtr ptr, double momentum);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetOptimizerNesterov(IntPtr ptr, double decay);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetOptimizerAdam(IntPtr ptr, double delta1, double delta2);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetDataSelectorFullBatch(IntPtr ptr);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetDataSelectorMiniBatch(IntPtr ptr, int batchSize);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetSchedulerConstant(IntPtr ptr);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetSchedulerIterationDecay(IntPtr ptr, double proportion);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetSchedulerExponentialDecay(IntPtr ptr, double decay);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetSchedulerInverseDecay(IntPtr ptr, double decay);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetSchedulerCosineDecay(IntPtr ptr, double endLearningRate, int iterationSpan);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void Predict(IntPtr ptr, 
        [In, Out, MarshalAs(UnmanagedType.LPArray)] double[] input, int inCount,
        [In, Out, MarshalAs(UnmanagedType.LPArray)] double[] output, int outCount);

    [LibraryImport("libExport.dll", StringMarshalling = StringMarshalling.Utf8)]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial IntPtr FromFile(string file);
    
    [LibraryImport("libExport.dll", StringMarshalling = StringMarshalling.Utf8)]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial int Save(IntPtr ptr, string file);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void Initialize(IntPtr ptr);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void Learn(IntPtr ptr, IntPtr learningState,
        [MarshalAs(UnmanagedType.LPArray)] double[] inputs, int inputCutOff, 
        [MarshalAs(UnmanagedType.LPArray)] double[] expected, int expectedCutOff, 
        int count, int iterations);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial double Cost(IntPtr ptr, double[] inputs, int inputCount, double[] expected,
        int expectedCount);

    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial double MultiCost(IntPtr ptr, double[] inputs, int inputCutOff, double[] expected,
        int expectedCutOff, int count);
}

public enum ActivationType
{
    DEFAULT,
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU,
    SILU,
    SOFTMAX,
}

public enum CostType
{
    MEAN_SQUARED,
    MEAN_ABSOLUTE,
    MEAN_LOG_COSH,
    BINARY_CROSS_ENTROPY
}