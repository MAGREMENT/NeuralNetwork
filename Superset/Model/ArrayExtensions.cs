using System.Globalization;

namespace Model;

public static class ArrayExtensions
{
    public static T[] Copy<T>(this T[] arr)
    {
        var result = new T[arr.Length];
        arr.CopyTo(result, 0);
        return result;
    }

    public static T[,] To2D<T>(this T[] arr, int width, int height)
    {
        var result = new T[width, height];

        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < height; j++)
            {
                result[i, j] = arr[j * width + i];
            }
        }

        return result;
    }

    public static void Shuffle<T>(this T[] array, int count)
    {
        var random = new Random();
        for (int i = 0; i < count; i++)
        {
            var to = random.Next(array.Length);

            (array[i], array[to]) = (array[to], array[i]);
        }
    }

    public static void Print<T>(this T[] array)
    {
        Console.Write("[");
        if (array.Length > 0)
        {
            Console.Write(array[0]);
            for (int i = 1; i < array.Length; i++)
            {
                Console.Write(", ");
                Console.Write(array[i]);
            }
        }
        Console.WriteLine("]");
    }
    
    public static void Print(this double[] array)
    {
        Print(array, array.Length);
    }
    
    public static void Print(this double[] array, int length)
    {
        var culture = CultureInfo.CreateSpecificCulture("en-US");
        
        Console.Write("[");
        if (length > 0)
        {
            Console.Write(array[0].ToString("0.0000", culture));
            for (int i = 1; i < length; i++)
            {
                Console.Write(", ");
                Console.Write(array[i].ToString("0.0000", culture));
            }
        }
        Console.WriteLine("]");
    }
}