#include "../../../thirdparty/TobiiProSDKWindows/64/include/tobii_research.h"
#include "../../../thirdparty/TobiiProSDKWindows/64/include/tobii_research_streams.h"
#include "../../../thirdparty/TobiiProSDKWindows/64/include/tobii_research.h"
#include "../../../thirdparty/zmq.h"
#include "../../../thirdparty/cJSON.h"
#include <windows.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int is_first_time = 1;
double time_offset = 0;
void *responder;

int64_t get_current_time_in_microseconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000 + (int64_t)tv.tv_usec;
}

int64_t calculate_time_offset(int64_t system_time_stamp) {
    int64_t current_time = get_current_time_in_microseconds();

    return current_time - system_time_stamp;
}

void gaze_data_callback(TobiiResearchGazeData *gaze_data, void *user_data) {
    memcpy(user_data, gaze_data, sizeof(*gaze_data));
    // TobiiResearchGazeData *data = (TobiiResearchEyeData *) user_data;
    float lgp2dx = gaze_data->left_eye.gaze_point.position_on_display_area.x;
    float lgp2dy = gaze_data->left_eye.gaze_point.position_on_display_area.y;
    float lgp3dx = gaze_data->left_eye.gaze_point.position_in_user_coordinates.x;
    float lgp3dy = gaze_data->left_eye.gaze_point.position_in_user_coordinates.y;
    float lgp3dz = gaze_data->left_eye.gaze_point.position_in_user_coordinates.z;
    int lgpv = gaze_data->left_eye.gaze_point.validity;
    float lpd = gaze_data->left_eye.pupil_data.diameter;
    int lpdv = gaze_data->left_eye.pupil_data.validity;
    float lgo3dux = gaze_data->left_eye.gaze_origin.position_in_user_coordinates.x;
    float lgo3duy = gaze_data->left_eye.gaze_origin.position_in_user_coordinates.y;
    float lgo3duz = gaze_data->left_eye.gaze_origin.position_in_user_coordinates.z;
    float lgo3dtx = gaze_data->left_eye.gaze_origin.position_in_track_box_coordinates.x;
    float lgo3dty = gaze_data->left_eye.gaze_origin.position_in_track_box_coordinates.y;
    float lgo3dtz = gaze_data->left_eye.gaze_origin.position_in_track_box_coordinates.z;
    int lgov = gaze_data->left_eye.gaze_origin.validity;

    float rgp2dx = gaze_data->right_eye.gaze_point.position_on_display_area.x;
    float rgp2dy = gaze_data->right_eye.gaze_point.position_on_display_area.y;
    float rgp3dx = gaze_data->right_eye.gaze_point.position_in_user_coordinates.x;
    float rgp3dy = gaze_data->right_eye.gaze_point.position_in_user_coordinates.y;
    float rgp3dz = gaze_data->right_eye.gaze_point.position_in_user_coordinates.z;
    int rgpv = gaze_data->right_eye.gaze_point.validity;
    float rpd = gaze_data->right_eye.pupil_data.diameter;
    int rpdv = gaze_data->right_eye.pupil_data.validity;
    float rgo3dux = gaze_data->right_eye.gaze_origin.position_in_user_coordinates.x;
    float rgo3duy = gaze_data->right_eye.gaze_origin.position_in_user_coordinates.y;
    float rgo3duz = gaze_data->right_eye.gaze_origin.position_in_user_coordinates.z;
    float rgo3dtx = gaze_data->right_eye.gaze_origin.position_in_track_box_coordinates.x;
    float rgo3dty = gaze_data->right_eye.gaze_origin.position_in_track_box_coordinates.y;
    float rgo3dtz = gaze_data->right_eye.gaze_origin.position_in_track_box_coordinates.z;
    int rgov = gaze_data->right_eye.gaze_origin.validity;

//    struct GazeData {
//        float lgp2dx, lgp2dy, lgp3dx, lgp3dy, lgp3dz;
//        int lgpv;
//        float lpd;
//        int lpdv;
//        float lgo3dux, lgo3duy, lgo3duz, lgo3dtx, lgo3dty, lgo3dtz;
//        int lgov;
//        float rgp2dx, rgp2dy, rgp3dx, rgp3dy, rgp3dz;
//        int rgpv;
//        float rpd;
//        int rpdv;
//        float rgo3dux, rgo3duy, rgo3duz, rgo3dtx, rgo3dty, rgo3dtz;
//        int rgov;
//    };
    float frames_array[] = {lgp2dx, rgp2dx, lgp2dy, rgp2dy, lgp3dx, rgp3dx, lgp3dy, rgp3dy, lgp3dz, rgp3dz, lgpv, rgpv, lpd, rpd, lpdv, rpdv, lgo3dux, rgo3dux, lgo3duy, rgo3duy, lgo3duz, rgo3duz, lgo3dtx, rgo3dtx, lgo3dty, rgo3dty, lgo3dtz, rgo3dtz, lgov, rgov};

    if (is_first_time) {
        time_offset = calculate_time_offset(gaze_data->system_time_stamp);
        is_first_time = 0;
    }

    double timestamps_array[] = {(double)(gaze_data->system_time_stamp + time_offset)};


    cJSON *root = cJSON_CreateObject();

    cJSON_AddStringToObject(root, "t", "d");
    cJSON *frames_json = cJSON_CreateFloatArray(frames_array, 30);
    cJSON_AddItemToObject(root, "frame", frames_json);
    cJSON *timestamps_json = cJSON_CreateDoubleArray(timestamps_array, 1);
    cJSON_AddItemToObject(root, "timestamp", timestamps_json);

    char *json_data = cJSON_Print(root);

    if (zmq_send(responder, json_data, strlen(json_data), 0) == -1) {
        fprintf(stderr, "Failed to send data via ZeroMQ.\n");
    } else {
        printf("Sent JSON Data: %s\n", json_data);
    }

    free(json_data);
    cJSON_Delete(root);

}

static void buffer_overflow_notification_callback(TobiiResearchNotification* notification, void* user_data) {
    (void)user_data;
    // Make sure we are handling the correct notification
    if (notification->notification_type == TOBII_RESEARCH_NOTIFICATION_STREAM_BUFFER_OVERFLOW) {
        //notification->value.text contains which streams' buffer is overflowing
        cJSON *root = cJSON_CreateObject();
        cJSON_AddStringToObject(root, "t", "e");
        cJSON_AddItemToObject(root, "message", "Buffer overflow occured.");
        //cJSON_AddItemToObject(root, "message", "Buffer overflow occured in %s stream buffer", notification->value.text);
    }
}

int main ()
{
    void *context = zmq_ctx_new();
    responder = zmq_socket(context, ZMQ_REP);
    int rc = zmq_bind(responder, "tcp://*:5555");

    TobiiResearchEyeTrackers* eyetrackers = NULL;

    TobiiResearchStatus result;
    size_t i = 0;

    result = tobii_research_find_all_eyetrackers(&eyetrackers);

    if (result != TOBII_RESEARCH_STATUS_OK) {
        printf("Finding trackers failed. Error: %d\n", result);
        return result;
    }

    TobiiResearchEyeTracker* eyetracker = eyetrackers->eyetrackers[0];
    char* address = NULL;
    char* serial_number = NULL;
    char* device_name = NULL;

    tobii_research_get_address(eyetracker, &address);
    tobii_research_get_serial_number(eyetracker, &serial_number);
    tobii_research_get_device_name(eyetracker, &device_name);

    printf("%s\t%s\t%s\n", address, serial_number, device_name);

    tobii_research_free_string(address);
    tobii_research_free_string(serial_number);
    tobii_research_free_string(device_name);

    tobii_research_free_eyetrackers(eyetrackers);

    TobiiResearchGazeData gaze_data;
    tobii_research_get_serial_number(eyetracker, &serial_number);

    printf("Subscribing to gaze data for eye tracker with serial number %s.\n", serial_number);

    tobii_research_free_string(serial_number);

    // Subscribe to notifications, passing a callback function defined above.
    TobiiResearchStatus status = tobii_research_subscribe_to_notifications(eyetracker, buffer_overflow_notification_callback, NULL);

    if (status != TOBII_RESEARCH_STATUS_OK)
        return 1;

    // Subscribe to gaze stream here. If your code takes to long too process the callback,
    // the circular buffer will overflow and buffer_overflow_notification_callback will be called.
    // For further information, we refer to our section on "streambufferoverflow".
    status = tobii_research_subscribe_to_gaze_data(eyetracker, gaze_data_callback, &gaze_data);

    if (status != TOBII_RESEARCH_STATUS_OK)
        return 1;

    /* Wait while some gaze data is collected. */
    Sleep(2000000);

    status = tobii_research_unsubscribe_from_gaze_data(eyetracker, gaze_data_callback);

    /* Wait while some gaze data is collected. */
    Sleep(2000000);
    tobii_research_unsubscribe_from_notifications(eyetracker, buffer_overflow_notification_callback);

    is_first_time = 1;
    zmq_close(responder);
    zmq_ctx_shutdown(context);

    return 0;

}