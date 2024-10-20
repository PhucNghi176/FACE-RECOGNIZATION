using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Microsoft.AspNetCore.Mvc;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddScoped<IFaceDetectionRepository, FaceDetectionRepository>();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddAntiforgery(options => { options.SuppressXFrameOptionsHeader = true; });

var app = builder.Build();
// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();
app.MapPost("api/v1/face-recognition", async (
    [FromServices] IFaceDetectionRepository faceDetectionRepository,
    [FromForm] IFormFile image,
    string employeeCode) =>
{
    try
    {
        var employeeFolderPath = Path.Combine(Directory.GetCurrentDirectory(), "TrainedFaces",
            short.Parse(employeeCode).ToString());
        var result = await faceDetectionRepository.SaveImageAndConvertToGrayScale(image, employeeCode, employeeFolderPath);
        return Results.Ok(result);
    }
    catch (Exception ex)
    {
        return Results.BadRequest($"Error processing request: {ex.Message}");
    }
}).AllowAnonymous().DisableAntiforgery();

app.MapPost("api/v1/face-recognition/detect", async (
    [FromServices] IFaceDetectionRepository faceDetectionRepository,
    [FromForm] IFormFile image) =>
{
    try
    {
        var directories = Directory.GetDirectories(Directory.GetCurrentDirectory() + "/TrainedFaces/");
        var result = await faceDetectionRepository.DetectFaceFromImage(image, directories);
        return Results.Ok(result);
    }
    catch (Exception ex)
    {
        return Results.BadRequest($"Error processing request: {ex.Message}");
    }
}).AllowAnonymous().DisableAntiforgery();
app.Run();
internal class FaceDetectionRepository : IFaceDetectionRepository
{
    private const string _unknownEmployeeFolder = "UnknownEmployeeFolder";
    private readonly string _cascadeFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "haarcascade_frontalface_default.xml");

    public async Task<Result<string>> DetectFaceFromImage(IFormFile image, string[] directories)
    {
        CascadeClassifier faceClassifier = new(_cascadeFilePath);
        var recognizedNames = "";
        var trainingImages = new List<Image<Gray, byte>>();
        var labels = new List<string>();

        try
        {
            // Load trained faces and labels
            LoadTrainingData(directories, trainingImages, labels);

            var filePath = await SaveImage(image);
            using var grayImage = new Image<Bgr, byte>(filePath.Value).Convert<Gray, byte>();

            // Detect faces
            var facesDetected = faceClassifier.DetectMultiScale(grayImage, 1.2, 10, new Size(20, 20), Size.Empty);

            foreach (var faceRect in facesDetected)
            {
                var result = grayImage.Copy(faceRect).Resize(100, 100, Inter.Cubic);
                if (trainingImages.Count == 0) continue;

                // Recognize face
                var recognizer = CreateRecognizer(trainingImages, labels);
                var prediction = recognizer.Predict(result);
                if (prediction.Label == -1) continue;

                recognizedNames = labels[prediction.Label];
            }

            return string.IsNullOrEmpty(recognizedNames) ? "No recognized faces" : $"Employee Code: {recognizedNames}";
        }
        catch (Exception ex)
        {
            throw new Exception($"Error during face detection: {ex.Message}");
        }
    }

    public async Task<Result<string>> SaveImageAndConvertToGrayScale(IFormFile image, string employeeId, string employeeFolderPath, CancellationToken cancellationToken = default)
    {
        if (!Directory.Exists(employeeFolderPath)) Directory.CreateDirectory(employeeFolderPath);

        if (image.Length == 0) return "Image is empty";

        var uniqueFileName = $"{Guid.NewGuid()}.bmp";
        var filePath = Path.Combine(employeeFolderPath, uniqueFileName);

        try
        {
            using var grayImage = await ConvertToGrayScale(image, cancellationToken);
            var faceRectangles = DetectFaces(grayImage);

            if (faceRectangles.Length == 0)
            {
                throw new Exception("No face detected in the image");
            }

            using var trainedFace = grayImage.Copy(faceRectangles[0]).Resize(100, 100, Inter.Cubic);
            trainedFace.Save(filePath);

            return filePath;
        }
        catch (Exception ex)
        {
            throw new Exception($"Error saving image: {ex.Message}");
        }
    }

    public async Task<Result<string>> SaveImage(IFormFile image, CancellationToken cancellationToken = default)
    {
        if (!Directory.Exists(_unknownEmployeeFolder)) Directory.CreateDirectory(_unknownEmployeeFolder);

        if (image.Length == 0) throw new Exception("Image is empty");

        var uniqueFileName = $"{Guid.NewGuid()}.bmp";
        var filePath = Path.Combine(_unknownEmployeeFolder, uniqueFileName);

        try
        {
            await using var stream = new FileStream(filePath, FileMode.Create);
            await image.CopyToAsync(stream, cancellationToken);
            return filePath;
        }
        catch (Exception ex)
        {
            throw new Exception($"Failed to save image: {ex.Message}");
        }
    }

    // Utility methods

    private static void LoadTrainingData(string[] directories, List<Image<Gray, byte>> trainingImages, List<string> labels)
    {
        foreach (var dir in directories)
        {
            var label = Path.GetFileNameWithoutExtension(dir);
            var files = Directory.GetFiles(dir, "*.bmp");

            foreach (var file in files)
            {
                trainingImages.Add(new Image<Gray, byte>(file));
                labels.Add(label);
            }
        }
    }

    private static EigenFaceRecognizer CreateRecognizer(List<Image<Gray, byte>> trainingImages, List<string> labels)
    {
        var recognizer = new EigenFaceRecognizer();
        using var imagesVector = new VectorOfMat(trainingImages.Select(img => img.Mat).ToArray());
        using var labelsVector = new VectorOfInt(Enumerable.Range(0, labels.Count).ToArray());

        recognizer.Train(imagesVector, labelsVector);
        return recognizer;
    }

    private static async Task<Image<Gray, byte>> ConvertToGrayScale(IFormFile image, CancellationToken cancellationToken)
    {
        using var memoryStream = new MemoryStream();
        await image.CopyToAsync(memoryStream, cancellationToken);
        memoryStream.Position = 0;

        using var mat = new Mat();
        CvInvoke.Imdecode(memoryStream.ToArray(), ImreadModes.Color, mat);
        return mat.ToImage<Bgr, byte>().Convert<Gray, byte>().Resize(320, 240, Inter.Cubic);
    }

    private static Rectangle[] DetectFaces(Image<Gray, byte> grayImage)
    {
        var faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml");
        return faceCascade.DetectMultiScale(grayImage, 1.2, 10, Size.Empty, Size.Empty);
    }
}


internal sealed class Error(string code, string message) : IEquatable<Error>
{
    public static readonly Error None = new(string.Empty, string.Empty);
    public static readonly Error NullValue = new("Error.NullValue", "The specified result value is null.");
    public static readonly Error UploadFail = new("Error.UploadFail", "Upload image fail.");
    private string Code { get; } = code;


    private string Message { get; } = message;

    public bool Equals(Error? other)
    {
        if (other is null) return false;

        return Code == other.Code && Message == other.Message;
    }

    public static implicit operator string(Error error)
    {
        return error.Code;
    }

    public static bool operator ==(Error? a, Error? b)
    {
        if (a is null && b is null) return true;

        if (a is null || b is null) return false;

        return a.Equals(b);
    }

    public static bool operator !=(Error? a, Error? b)
    {
        return !(a == b);
    }

    public override bool Equals(object? obj)
    {
        return obj is Error error && Equals(error);
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(Code, Message);
    }

    public override string ToString()
    {
        return Code;
    }
}

internal class Result
{
    protected Result(bool isSuccess, Error error)
    {
        switch (isSuccess)
        {
            case true when error != Error.None:
                throw new InvalidOperationException();
            case false when error == Error.None:
                throw new InvalidOperationException();
            default:
                IsSuccess = isSuccess;
                Error = error;
                break;
        }
    }

    protected bool IsSuccess { get; }

    public bool IsFailure => !IsSuccess;

    public Error Error { get; }

    public static Result Success()
    {
        return new Result(true, Error.None);
    }

    private static Result<TValue> Success<TValue>(TValue value)
    {
        return new Result<TValue>(value, true, Error.None);
    }

    public static Result Failure(Error error)
    {
        return new Result(false, error);
    }

    private static Result<TValue> Failure<TValue>(Error error)
    {
        return new Result<TValue>(default, false, error);
    }

    protected static Result<TValue> Create<TValue>(TValue? value)
    {
        return value is not null ? Success(value) : Failure<TValue>(Error.NullValue);
    }
}

internal class Result<TValue> : Result
{
    private readonly TValue? _value;

    protected internal Result(TValue? value, bool isSuccess, Error error)
        : base(isSuccess, error)
    {
        _value = value;
    }

    public TValue Value => IsSuccess
        ? _value!
        : throw new InvalidOperationException("The value of a failure result can not be accessed.");

    public static implicit operator Result<TValue>(TValue? value)
    {
        return Create(value);
    }
}

internal interface IFaceDetectionRepository
{
    Task<Result<string>> DetectFaceFromImage(IFormFile image, string[] directories);

    Task<Result<string>> SaveImageAndConvertToGrayScale(IFormFile image, string employeeId, string employeeFolderPath,
        CancellationToken cancellationToken = default);
    Task<Result<string>> SaveImage(IFormFile image,
        CancellationToken cancellationToken = default);
}