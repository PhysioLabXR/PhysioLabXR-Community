from multiprocessing import Manager, Process

from physiolabxr.presets.Presets import Presets


def child_process(shared_namespace):
    presets = shared_namespace.presets
    presets._preset_root = 'child_process_name'
    print(f"Child process presets value: {presets._preset_root}")

if __name__ == '__main__':
    presets = Presets(_preset_root='../_presets', _reset=False)  # create the singleton presets object

    manager = Manager()
    shared_namespace = manager.Namespace()
    shared_namespace.presets = presets

    print(f"Before running the child process: Parent process presets value: {presets._preset_root}")

    p = Process(target=child_process, args=(shared_namespace,))
    p.start()
    p.join()

    print(f"After running the child process: Parent process presets value: {presets._preset_root}")
    manager.shutdown()