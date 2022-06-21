'''
main window button control
'''

# signal_settings_btn.clicked.connect(signal_settings_window)
#### TODO: signal processing button (hidded before finishing)
# signal_settings_btn.hide()

#####
# pop window actions
# def dock_window():
#     self.sensorTabSensorsHorizontalLayout.insertWidget(self.sensorTabSensorsHorizontalLayout.count() - 1,
#                                                        lsl_stream_widget)
#     pop_window_btn.clicked.disconnect()
#     pop_window_btn.clicked.connect(pop_window)
#     pop_window_btn.setText('Pop Window')
#     self.pop_windows[lsl_stream_name].hide()  # tetentive measures
#     self.pop_windows.pop(lsl_stream_name)
#     lsl_stream_widget.set_button_icons()
#
# def pop_window():
#     w = AnotherWindow(lsl_stream_widget, remove_stream)
#     self.pop_windows[lsl_stream_name] = w
#     w.setWindowTitle(lsl_stream_name)
#     pop_window_btn.setText('Dock Window')
#     w.show()
#     pop_window_btn.clicked.disconnect()
#     pop_window_btn.clicked.connect(dock_window)
#     lsl_stream_widget.set_button_icons()
#
# pop_window_btn.clicked.connect(pop_window)
#
# def start_stop_stream_btn_clicked():
#     # check if is streaming
#     if self.lsl_workers[lsl_stream_name].is_streaming:
#         self.lsl_workers[lsl_stream_name].stop_stream()
#         if not self.lsl_workers[lsl_stream_name].is_streaming:
#             # started
#             print("sensor stopped")
#             # toggle the icon
#             start_stop_stream_btn.setText("Start Stream")
#     else:
#         self.lsl_workers[lsl_stream_name].start_stream()
#         if self.lsl_workers[lsl_stream_name].is_streaming:
#             # started
#             print("sensor stopped")
#             # toggle the icon
#             start_stop_stream_btn.setText("Stop Stream")
#     lsl_stream_widget.set_button_icons()
#
#
# start_stop_stream_btn.clicked.connect(start_stop_stream_btn_clicked)

# def remove_stream():
#     if self.recording_tab.is_recording:
#         dialog_popup(msg='Cannot remove stream while recording.')
#         return False
#     # stop_stream_btn.click()  # fire stop streaming first
#     if self.lsl_workers[lsl_stream_name].is_streaming:
#         self.lsl_workers[lsl_stream_name].stop_stream()
#     worker_thread.exit()
#     self.lsl_workers.pop(lsl_stream_name)
#     self.worker_threads.pop(lsl_stream_name)
#     # if this lsl connect to a device:
#     if lsl_stream_name in self.device_workers.keys():
#         self.device_workers[lsl_stream_name].stop_stream()
#         self.device_workers.pop(lsl_stream_name)
#
#     self.stream_ui_elements.pop(lsl_stream_name)
#     self.sensorTabSensorsHorizontalLayout.removeWidget(lsl_stream_widget)
#     # close window if popped
#     if lsl_stream_name in self.pop_windows.keys():
#         self.pop_windows[lsl_stream_name].hide()
#         self.pop_windows.pop(lsl_stream_name)
#     else:  # use recursive delete if docked
#         sip.delete(lsl_stream_widget)
#     self.LSL_data_buffer_dicts.pop(lsl_stream_name)
#     return True
#
# #     worker_thread
# remove_stream_btn.clicked.connect(remove_stream)

# remove_stream_btn = init_button(parent=lsl_layout, label='Remove Stream',
#                                 function=remove_stream)  # add delete sensor button after adding visualization
