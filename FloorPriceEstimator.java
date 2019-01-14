package com.mobfox.auctioncore.logic;

import com.google.protobuf.ByteString;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.example.*;

import java.nio.charset.Charset;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;

// We are binding an instance of this class as a Singleton using Guice injection 
// bind(FloorPriceEstimator.class).in(Singleton.class);
public class FloorPriceEstimator {

    private static final String MODEL_PATH = "/src/main/resources/model/1546946605";
    final Path path = Paths.get(Paths.get("").toAbsolutePath().toString(), MODEL_PATH);
    // Make sure to call model.close() if used differently
    final SavedModelBundle model = SavedModelBundle.load(path.toString(), "serve");

    public Optional<Float> predict(String inventoryId, String requestType, Float exFloorPrice, String stateCode,
      String countryCode, String cityCode, String deviceOs, String deviceOsVersion, String hourOfDay)
    {
        Example exampleInput = constructInput(inventoryId, requestType, exFloorPrice, stateCode, countryCode, cityCode,
          deviceOs, deviceOsVersion, hourOfDay);

        //requires an array of inputs
        byte[][] inputs = new byte[1][];
        inputs[0] = exampleInput.toByteArray();

        Float result = null;
        try(
          Tensor<?> in = Tensor.create(inputs);
          Tensor<?> predictions = model.session().runner()
              .feed("input_example_tensor", in)
              .fetch("Squeeze:0").run().get(0))
        {
            result = predictions.copyTo(new float[1])[0];
        } catch (Exception e) {}

        return Optional.ofNullable(result);
    }

    private Example constructInput(String inventoryId, String requestType, Float exFloorPrice, String stateCode,
      String countryCode, String cityCode, String deviceOs, String deviceOsVersion, String hourOfDay)
    {
        Features features = Features.newBuilder()
          .putFeature("inventory_id", createFeature(inventoryId))
          .putFeature("request_type", createFeature(requestType))
          .putFeature("ex_floor_price", createFeature(exFloorPrice))
          .putFeature("state_code", createFeature(stateCode))
          .putFeature("country_code", createFeature(countryCode))
          .putFeature("city_code", createFeature(cityCode))
          .putFeature("device_os", createFeature(deviceOs))
          .putFeature("device_os_version", createFeature(deviceOsVersion))
          .putFeature("hour_of_day", createFeature(hourOfDay))
          .build();
        // This will build a proto example from raw data for the inference
        return Example.newBuilder().setFeatures(features).build();
    }

    private Feature createFeature(String value) {
        return Feature.newBuilder().setBytesList(
          BytesList.newBuilder().addValue(ByteString.copyFrom(value, Charset.forName("UTF-8"))).build())
          .build();
    }

    private Feature createFeature(Float value) {
        return Feature.newBuilder().setFloatList(
          FloatList.newBuilder().addValue(value).build())
          .build();
    }
}
