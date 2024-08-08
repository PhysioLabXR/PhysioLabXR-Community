import grpc
import example_pb2
import asyncio
import tkinter as tk
from tkinter import ttk
from concurrent.futures import ThreadPoolExecutor

from physiolabxr.examples.rpc.Async import AsyncRPCExample_pb2_grpc, AsyncRPCExample_pb2


class AsyncClient:
    def __init__(self):
        self.loop = asyncio.get_event_loop()
        self.executor = ThreadPoolExecutor()
        self.channel = grpc.aio.insecure_channel('localhost:13004')
        self.stub = AsyncRPCExample_pb2_grpc.AsyncRPCExampleStub(self.channel)

    async def send_request(self, message):
        request = AsyncRPCExample_pb2.AsyncOneArgOneReturnRequest(input0=message)
        response = await self.stub.AsyncOneArgOneReturn(request)
        return response.message

    def run_async(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def start(self):
        self.loop.run_in_executor(self.executor, self.loop.run_forever)

class App:
    def __init__(self, root, client):
        self.client = client
        self.root = root
        self.root.title("Async gRPC Client")
        self.label = ttk.Label(root, text="Enter your message:")
        self.label.pack(padx=10, pady=10)
        self.entry = ttk.Entry(root)
        self.entry.pack(padx=10, pady=10)
        self.button = ttk.Button(root, text="Send", command=self.on_send)
        self.button.pack(padx=10, pady=10)
        self.result = ttk.Label(root, text="")
        self.result.pack(padx=10, pady=10)

    def on_send(self):
        message = self.entry.get()
        self.result.config(text="Sending request...")
        future = self.client.run_async(self.client.send_request(message))
        future.add_done_callback(self.on_response)

    def on_response(self, future):
        response = future.result()
        self.result.config(text=f"Received response: {response}")

if __name__ == "__main__":
    root = tk.Tk()
    client = AsyncClient()
    client.start()
    app = App(root, client)
    root.mainloop()
