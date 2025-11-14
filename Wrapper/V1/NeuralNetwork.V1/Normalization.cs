using System.Runtime.InteropServices;
using Base;

namespace NeuralNetwork.V1;

public static partial class Normalization
{
    public static void Standardize(FlattenedData data)
    {
        Standardize(data.Inputs, data.InputCutOff, data.Expected, data.ExpectedCutOff, data.GetCount());
    }
    
    public static void MinMaxScale(FlattenedData data)
    {
        MinMaxScale(data.Inputs, data.InputCutOff, data.Expected, data.ExpectedCutOff, data.GetCount());
    }
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void Standardize(
        [MarshalAs(UnmanagedType.LPArray)] double[] inputs, int inputCutOff, 
        [MarshalAs(UnmanagedType.LPArray)] double[] expected, int expectedCutOff, 
        int count);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void MinMaxScale(
        [MarshalAs(UnmanagedType.LPArray)] double[] inputs, int inputCutOff, 
        [MarshalAs(UnmanagedType.LPArray)] double[] expected, int expectedCutOff, 
        int count);
}