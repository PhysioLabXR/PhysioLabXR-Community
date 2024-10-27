from PyQt6.QtCore import QThread

from physiolabxr.sub_process.TCPInterface import RenaTCPInterface
from physiolabxr.threadings import workers


# def create_worker(self):
#     self.stdout_socket_interface = RenaTCPInterface(stream_name='RENA_SCRIPTING_STDOUT',
#                                                     port_id=self.port,
#                                                     identity='client',
#                                                     pattern='router-dealer')
#     self.stdout_worker_thread = QThread(self.parent)
#     self.stdout_worker = workers.ScriptingStdoutWorker(self.stdout_socket_interface)
#     self.stdout_worker.std_signal.connect(self.redirect_script_std)
#     self.stdout_worker.moveToThread(self.stdout_worker_thread)
#     self.stdout_worker_thread.start()
#     self.stdout_timer = QTimer()
#     self.stdout_timer.setInterval(SCRIPTING_UPDATE_REFRESH_INTERVAL)
#     self.stdout_timer.timeout.connect(self.stdout_worker.tick_signal.emit)
#     self.stdout_timer.start()