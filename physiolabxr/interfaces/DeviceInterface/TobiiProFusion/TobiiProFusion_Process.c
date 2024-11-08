#include "tobii_research.h"
#include "tobii_research_streams.h"
#include "tobii_research_eyetracker.h"
#include "cJSON.h"
#include <zmq.h>

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <assert.h>

int is_first_time = 1;
int64_t time_offset = 0;
void* context = NULL;
TobiiResearchEyeTracker* eyetracker = NULL;
void* tobii_socket;

int64_t get_current_time_in_microseconds() {
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    ULARGE_INTEGER time;
    time.LowPart = ft.dwLowDateTime;
    time.HighPart = ft.dwHighDateTime;
    return (int64_t)(time.QuadPart / 10);
}

int64_t calculate_time_offset(int64_t system_time_stamp) {
    int64_t current_time = get_current_time_in_microseconds();

    return current_time - system_time_stamp;
}

void gaze_data_callback(TobiiResearchGazeData* gaze_data, void* user_data) {
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

    float frames_array[] = { lgp3dx, rgp3dx, lgp3dy, rgp3dy, lgp3dz, rgp3dz, lpd, rpd, lgp2dx, rgp2dx, lgp2dy, rgp2dy };

    if (is_first_time) {
        time_offset = calculate_time_offset(gaze_data->system_time_stamp);
        is_first_time = 0;
    }

    double timestamps_array[] = { (double)(gaze_data->system_time_stamp + time_offset) };


    cJSON* root = cJSON_CreateObject();

    cJSON_AddStringToObject(root, "t", "d");
    cJSON* frames_json = cJSON_CreateFloatArray(frames_array, 30);
    cJSON_AddItemToObject(root, "frame", frames_json);
    cJSON* timestamps_json = cJSON_CreateDoubleArray(timestamps_array, 1);
    cJSON_AddItemToObject(root, "timestamp", timestamps_json);

    char* json_data = cJSON_Print(root);

    if (zmq_send(tobii_socket, json_data, strlen(json_data), ZMQ_DONTWAIT) == -1) {
        fprintf(stderr, "Failed to send data via ZeroMQ.\n");
    }
    else {
    }

    free(json_data);
    cJSON_Delete(root);

}

void handle_signal(int signal) {
    if (signal == SIGTERM) {
        printf("Received SIGTERM, cleaning up resources...\n");

        // Unsubscribe from gaze data if applicable
        if (eyetracker) {
            tobii_research_unsubscribe_from_gaze_data(eyetracker, gaze_data_callback);
        }

        // Close and terminate ZeroMQ resources
        if (tobii_socket) {
            zmq_close(tobii_socket);
            tobii_socket = NULL; // Avoid dangling pointer
        }
        if (context) {
            zmq_ctx_term(context);
            context = NULL; // Avoid dangling pointer
        }

        // Perform any additional cleanup
        printf("Resources cleaned up successfully.\n");
        exit(0); // Ensure program exits after cleanup
    }
}

static void buffer_overflow_notification_callback(TobiiResearchNotification* notification, void* user_data) {
    (void)user_data;
    // Make sure we are handling the correct notification
    if (notification->notification_type == TOBII_RESEARCH_NOTIFICATION_STREAM_BUFFER_OVERFLOW) {
        //notification->value.text contains which streams' buffer is overflowing
        cJSON* root = cJSON_CreateObject();
        cJSON_AddStringToObject(root, "t", "e");
        cJSON_AddStringToObject(root, "message", "Buffer overflow occured.");
        //cJSON_AddItemToObject(root, "message", "Buffer overflow occured in %s stream buffer", notification->value.text);
    }
}


void tobii_pro_fusion_process(int port) {
    signal(SIGTERM, handle_signal);
    printf("Entered tobii_pro_fusion_process with port %d\n", port);

    // Initialize ZMQ context and socket
    char endpoint[50];
    sprintf_s(endpoint, sizeof(endpoint), "tcp://localhost:%d", port);

    context = zmq_ctx_new();
    if (!context) {
        fprintf(stderr, "Failed to create ZMQ context: %s\n", zmq_strerror(zmq_errno()));
        return;
    }
    tobii_socket = zmq_socket(context, ZMQ_PUSH);
    int rc = zmq_connect(tobii_socket, endpoint);
    if (rc != 0) {
        fprintf(stderr, "Failed to bind ZMQ socket: %s\n", zmq_strerror(zmq_errno()));
        return;
    }
    else {
        printf("ZMQ socket bound successfully on port %d\n", port);
    }


    // Initialize Tobii Research
    TobiiResearchEyeTrackers* eyetrackers = NULL;
    TobiiResearchStatus result = tobii_research_find_all_eyetrackers(&eyetrackers);
    if (result != TOBII_RESEARCH_STATUS_OK) {
        printf("Finding trackers failed. Error: %d\n", result);
        return;
    }

    eyetracker = eyetrackers->eyetrackers[0];
    char* address = NULL, * serial_number = NULL, * device_name = NULL;
    tobii_research_get_address(eyetracker, &address);
    tobii_research_get_serial_number(eyetracker, &serial_number);
    tobii_research_get_device_name(eyetracker, &device_name);
    printf("Connected to eye tracker at %s, serial: %s, device: %s\n", address, serial_number, device_name);

    tobii_research_free_string(address);
    tobii_research_free_string(serial_number);
    tobii_research_free_string(device_name);
    tobii_research_free_eyetrackers(eyetrackers);

    // Subscribe to gaze data
    TobiiResearchGazeData gaze_data;
    TobiiResearchStatus status = tobii_research_subscribe_to_gaze_data(eyetracker, gaze_data_callback, &gaze_data);
    if (status != TOBII_RESEARCH_STATUS_OK) {
        printf("Error subscribing to gaze data: %d\n", status);
        return;
    }

    printf("Started gaze data collection on port %d\n", port);

    // Continuously collect and send data
    while (1) {
        Sleep(4000);
    }

    // Clean up when finished
    tobii_research_unsubscribe_from_gaze_data(eyetracker, gaze_data_callback);
    zmq_close(tobii_socket);
    zmq_ctx_term(context);
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        printf("Usage: %s <port>\n", argv[0]);
        return 1;
    }

    int port = atoi(argv[1]);
    printf("Program started with port: %d\n", port);
    tobii_pro_fusion_process(port);
    return 0;
}
