---
title: gRPC 源码解析
date: 2021-01-08 04:07:49
tags: 
    - gRPC
    - 源码解析
categories:
    - 系统架构
    - 源码解析
---


gRPC 重要概念源码

[toc]

# 通用概念



gRPC 中, 有部分概念在服务端和客户端由类似的定义, 比如:

* 拦截器 Interceptor
* 上下文 Context



## 拦截器 Interceptor

拦截器是在各种网络框架中经常使用的概念, 主要用于对网络请求进行前置和后置操作.

gRPC 的客户端和服务端都有自己的拦截器.  



### 服务端拦截器

请参考 [服务端概念->服务端拦截器]()



### 客户端拦截器

请参考 [客户端概念->客户端拦截器]()



## 上下文 Context

gRPC 提供了三种类型的 Context:

* 共享 Context `grpc.RpcContext`
* 服务端 Context `grpc.ServicerContext`
* 客户端 Context `grpc.Call`



### 共享 Context

共享 Context 接口定义为 `grpc.RpcContext` , 该接口定义如下:

```python
class RpcContext(six.with_metaclass(abc.ABCMeta)):
    def is_active(self):
        pass

    def time_remaining(self):
        pass

    def cancel(self):
        pass

    def add_callback(self, callback):
        pass
```



服务端上下文 `grpc.ServicerContext`和客户端上下文 `grpc.Call` 均为该类的子类. 



### 服务端 Context

请参考 [服务端概念->服务端上下文]()



### 客户端 Context

请参考 [客户端概念->客户端上下文]()



# 服务端概念

服务端的基本概念有:

* 服务端 Server
* 服务 Servicer
* 接口 Handler
* 服务拦截器 Interceptor
* `ServicerContext`



## 服务端 Server



### 创建服务 `grpc.server()`

```python
# grpc/__init__.py

def server(
	thread_pool: futures.ThreadPoolExecutor,
	handlers: List[GenericRpcHandler] = None,
	interceptors: List[ServerInterceptor] = None,
	options: list[tuple] = None,
	maximum_concurrent_rpcs: int = None,
	compression: grpc.Compression = None,
):

    """Creates a Server with which RPCs can be serviced.

    Args:
      thread_pool: A futures.ThreadPoolExecutor to be used by the Server
        to execute RPC handlers.
      handlers: An optional list of GenericRpcHandlers used for executing RPCs.
        More handlers may be added by calling add_generic_rpc_handlers any time
        before the server is started.
      interceptors: An optional list of ServerInterceptor objects that observe
        and optionally manipulate the incoming RPCs before handing them over to
        handlers. The interceptors are given control in the order they are
        specified. This is an EXPERIMENTAL API.
      options: An optional list of key-value pairs (:term:`channel_arguments` in gRPC runtime)
        to configure the channel.
      maximum_concurrent_rpcs: The maximum number of concurrent RPCs this server
        will service before returning RESOURCE_EXHAUSTED status, or None to
        indicate no limit.
      compression: An element of grpc.compression, e.g.
        grpc.compression.Gzip. This compression algorithm will be used for the
        lifetime of the server unless overridden. This is an EXPERIMENTAL option.

    Returns:
      A Server object.
    """  
    pass
```

### Server 接口 `grpc.Server`

`grpc.Server` 抽象类定义如下

```python
class Server(six.with_metaclass(abc.ABCMeta)):
    def add_generic_rpc_handlers(self, generic_rpc_handlers):
        pass

    def add_insecure_port(self, address):
        pass

    def add_secure_port(self, address, server_credentials):
        pass

    def start(self):
        pass

    def stop(self, grace):
        pass

    def wait_for_termination(self, timeout=None):
        pass
```

### Cython 实现 `cygrpc.Server`

其具体实现在 `src/python/grpcio/grpc/_server.py#_Server` 中.  而 `_Server` 类则调用了`src/python/grpcio/grpc/_cython/_cygrpc/server.pyx.pxi` 中使用 [Cython](https://cython.org/) 实现的 `cygrpc.Server` 类.

`cygrpc.Server` 的初始化函数定义如下:

```cython
  def __cinit__(self, object arguments):
    fork_handlers_and_grpc_init()
    self.references = []
    self.registered_completion_queues = []
    self.is_started = False
    self.is_shutting_down = False
    self.is_shutdown = False
    self.c_server = NULL
    cdef _ChannelArgs channel_args = _ChannelArgs(arguments)
    self.c_server = grpc_server_create(channel_args.c_args(), NULL)
    self.references.append(arguments)
```



### CPP 实现 `grpc_server` 与 `grpc_core::Server`

`cython.Server` 调用 C 函数 `grpc_server_create` 创建了 `grpc_server`,  `grpc_server` 是对 `grpc_core::Server` 的封装, 后者是 gRPC 中 server 的最终实现.



`grpc_server_create` 函数在 `include/grpc/grpc.h` 中声明, 在 `src/core/lib/surface/server.cc` 文件实现.

`grpc_server` 则定义在 `src/core/lib/surface/server.h` 文件中.

`grpc_core::Server` 在 `src/core/lib/surface/server.h` 定义, 大部分成员函数在 `src/core/lib/surface/server.cc` 中实现.



## 服务 Servicer

当我们编译 `.proto` 文件时, `grpcio_tools.protoc` 中的 gRPC 编译器扩展会基于 `.proto` 中的 `service` 定义自动生成 `XXXServicer` 抽象类 以及对应的辅助函数 `add_XXXServicer_to_server(servicer: XXXServicer, server: grpc.Server)`.



`XXXServicer` 与 `.proto` 中的 `service` 定义一致, 我们重点关注其中的接口函数.

以较为复杂的双向流接口为例:

`.proto` 文件中定义如下:

```protobuf
// 文件名 examples/protos/route_guide.proto
service examples/python/route_guide/route_guide_pb2_grpc.py {
  // 省略其它内容
  
  // A Bidirectional streaming RPC.
  //
  // Accepts a stream of RouteNotes sent while a route is being traversed,
  // while receiving other RouteNotes (e.g. from other users).
  rpc RouteChat(stream RouteNote) returns (stream RouteNote) {}
}
```



生成的 `Servicer` 定义如下:

```python
# 文件名 examples/python/route_guide/route_guide_pb2_grpc.py
class RouteGuideServicer(object):

  def RouteChat(
      self, 
      request_iterator: grpc._server._RequestIterator, 
      context: grpc.ServicerContext,
  ):
    """A Bidirectional streaming RPC.

    Accepts a stream of RouteNotes sent while a route is being traversed,
    while receiving other RouteNotes (e.g. from other users).
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')
```



辅助函数则是对 `grpc.Server.add_generic_rpc_handlers` 的封装, 其定义如下:

```python
# 文件名 examples/python/route_guide/route_guide_pb2_grpc.py
def add_RouteGuideServicer_to_server(servicer: RouteGuideServicer, server: grpc.Server):
  rpc_method_handlers = {
      # 省略其它内容
      'RouteChat': grpc.stream_stream_rpc_method_handler(
          servicer.RouteChat,
          request_deserializer=route__guide__pb2.RouteNote.FromString,
          response_serializer=route__guide__pb2.RouteNote.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler('routeguide.RouteGuide', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
```

上面的 `generic_handler` 是一个 `grpc._utilities.DictionaryGenericHandler` 对象, 其属性 `_method_handlers` 是一个字典, key 为 `/service/method` (如上述 `/routeguide.RouteGuide/RouteChat`), 其 value 是 `grpc._utilities.RpcMethodHandler` 对象.

 

## 接口 Handler

gRPC 提供了两种 Handler 接口:

* `grpc.RpcMethodHandler` 方法 Handler
* `grpc.ServiceRpcHandler` 服务 Handler

方法 Handler 和 服务 Handler 一般由 gRPC 编译器  `grpc_tools.protoc` 自动生成, 无需我们自行创建.

### 方法 Handler

方法 Handler 的接口定义为  `grpc.RpcMethodHandler`, 具体实现则为 `grpc._utilities.RpcMethodHandler`.

其继承关系图如下

```
grpc._utilities.RpcMethodHandler 
	-> grpc.RpcMethodHandler 
		-> abc.ABCMeta (metaclass)                         
    -> collections.namedtuple
```



gRPC 提供了四种创建 `grpc.RpcMethodHandler` 的方法:

* `unary_unary_rpc_method_handler`
* `unary_stream_rpc_method_handler`
* `stream_unary_rpc_method_handler`
* `stream_stream_rpc_method_handler`

这些方法与 gRPC 提供的四种接口类型一一对应.  这些方法拥有相似的函数签名,

```python
def xxx_yyy_rpc_method_handler(
    behavior: Callable, 
    request_deserializer=None, 
    response_serializer=None,
):
    pass
```



其中 `behavior` 是生成的 Servicer 中对应 `.proto` 中的 RPC 方法, `request_deserializer` 是 RPC 方法的输入参数 request 的反序列化方法,  `response_serialzer` RPC 方法的输出 response 的 序列化方法.



 `grpc._utilities.RpcMethodHandler ` 的定义如下:

```python
class RpcMethodHandler(
        collections.namedtuple('_RpcMethodHandler', (
            'request_streaming',
            'response_streaming',
            'request_deserializer',
            'response_serializer',
            'unary_unary',
            'unary_stream',
            'stream_unary',
            'stream_stream',
        )), grpc.RpcMethodHandler):
    pass
```



其父类 `grpc.RpcMethodHandler` 没有定义自己的方法和属性, 因此其所有该类有 8 个属性:

* `request_streaming`, `bool` 变量, 表明请求是否为流式
* `response_streaming`, `bool` 变量, 表明响应是否为流式
* `request_deserializer`, 请求序列化方法 (在编译时自动生成)
* `response_serializer`, 响应序列化方法 (在编译时自动生成)
* `unary_unary`, 响应函数
* `unary_stream`, 响应函数
* `stream_unary`, 响应函数
* `stream_stream`, 响应函数

不同类型的 handler 只需要赋值不同的属性即可, 如  `stream_stream_rpc_method_handler` 类 handler, 其初始化方法为:

```python
_utilities.RpcMethodHandler(
	request_streaming=True, 
	response_streaming=True, 
	request_deserializer=request_deserializer,
	response_serializer=response_serializer, 
	unary_unary=None, 
	unary_stream=None, 
	stream_unary=None,
	stream_stream=behavior,
)
```



### 服务 Handler

服务 Handler 是一种特殊的 Handler, 可以将其理解为同一服务的各个方法 Handler 组成的集合.



服务 Handler 的接口定义为 `grpc.ServiceRpcHandler`, 该接口是 `grpc.GenericRpcHandler` 的子接口. 具体实现则为 `grpc._utilities.DictionaryGenericHandler`. 



继承关系图如下:

```
grpc._utilities.DictionaryGenericHandler 
	-> grpc.ServiceRpcHandler 
		-> abc.ABCMeta (metaclass)
        -> grpc.GenericRpcHandler (metaclass)
        	-> abc.ABCMeta (metaclass)
```



可以使用 gRPC 提供的 `grpc.method_handlers_generic_handler` 创建服务 Handler.



`grpc._utilities.DictionaryGenericHandler` 的定义如下:

```python
class DictionaryGenericHandler(grpc.ServiceRpcHandler):

    def __init__(
        self, 
        service: str, 
        method_handlers: Mapping[str, grpc.RpcMethodHandler],
    ):
        self._name = service
        self._method_handlers = {
            _common.fully_qualified_method(service, method): method_handler
            for method, method_handler in six.iteritems(method_handlers)
        }

    def service_name(self) -> str:
        return self._name

    def service(self, handler_call_details: grpc.HandlerCallDetails) -> grpc.RpcMethodHandler:
        return self._method_handlers.get(handler_call_details.method)
```



其中 `service_name()` 是从 `grpc.ServiceRpcHandler` 继承的方法, `service(handler_call_details: grpc.HandlerCallDetails)` 是从 `grpc.GenericRpcHandler` 类继承的方法.



## 服务端拦截器

服务端拦截器的接口定义为 `grpc.ServerInterceptor`, 主要提供了一个方法 `intercept_service`, 该方法签名如下:

```python
def intercept_service(
    self, 
    # continuation 是一个函数, 该函数接收一个 grpc.HandlerCallDetails 参数
    continuation: Callable, 
    handler_call_details: grpc.HandlerCallDetails,
):
    pass
```



其中参数 `handler_call_details` 记录了一个方法的的基本信息, 如:

* `method` 方法名
* `invocation_metadata` 客户端调用时传入的元数据
* 等



gRPC 没有提供内置的服务端拦截器实现, 具体实现可以参考:

* `examples/python/interceptors/headers/request_header_validator_interceptor.py`
* `examples/python/auth/customized_auth_server.py`
* `src/python/grpcio_tests/tests/unit/_interceptor_test.py`



一个服务端拦截器实现示例如下:

```python
# 文件: src/python/grpcio_tests/tests/unit/_interceptor_test.py

class _GenericServerInterceptor(grpc.ServerInterceptor):

    def __init__(self, fn):
        self._fn = fn

    def intercept_service(self, continuation, handler_call_details):
        return self._fn(continuation, handler_call_details)


def _filter_server_interceptor(condition, interceptor):

    def intercept_service(continuation, handler_call_details):
        if condition(handler_call_details):
            return interceptor.intercept_service(continuation,
                                                 handler_call_details)
        return continuation(handler_call_details)

    return _GenericServerInterceptor(intercept_service)
```





## 服务端上下文



服务端上下文接口定义为 `grpc.ServicerContext`, 该类继承了共享上下文 `grpc.RpcContext`.

`grpc.ServicerContext` 在 `grpc.RpcContext` 的基础上新增了 13 个服务接口.

```python
class ServicerContext(six.with_metaclass(abc.ABCMeta, RpcContext)):
    def invocation_metadata(self):
        pass

    def peer(self):
        pass

    def peer_identities(self):
        pass

    def peer_identity_key(self):
        pass

    def auth_context(self):
        pass

    def set_compression(self, compression):
        pass

    def send_initial_metadata(self, initial_metadata):
        pass

    def set_trailing_metadata(self, trailing_metadata):
        pass

    def abort(self, code, details):
        pass

    def abort_with_status(self, status):
        pass

    def set_code(self, code):
        pass

    def set_details(self, details):
        pass

    def disable_next_mssage_compression(self):
        pass
```



根据 gRPC 的设计理念, gRPC 服务中的异常都建议使用 `grpc.ServicerContext` 回传给调用方, 而不是使用自定义响应包中的 `error_code` 和 `error_message`.



`grpc.ServicerContext` 中有一个重要方法 `set_code()`, 该方法可以设置一个 `grpc.StatusCode`, 并最终由客户端上下文 `grpc.Call` 中的 `code()` 方法返回给调用者.

状态码 `grpc.StatusCode` 是一个枚举类型, 定义了 gRPC 工作过程中常用的状态码, 如 `OK`, `CANCELLED`, `RESOURCE_EXHAUSTED` 等. 具体可以参考 [gRPC Status Code 官方文档](https://grpc.github.io/grpc/python/grpc.html#grpc-status-code).

服务端 Context 的具体实现为 `grpc._server._Context`, 该类定义在 `src/python/grpcio/grpc/_server.py` 文件中, 在此不详细展开.



# 客户端概念

客户端的主要概念有:

* Stub
* Channel
* 接口 Callable
* 客户端拦截器 Interceptor
* 客户端上下文 Context
* 异步调用结果 Future



## 客户端 Stub

客户端 Stub 是编译 `.proto` 文件时自动生成的. Stub 初始化时需要传入 Channel, 对 Stub 的所有请求最终都会由 Channel 来完成.

还是以前面的 `RouteGuide` 为例, 生成的 `RouteGuideStub` 如下:

```python
# 文件名 examples/python/route_guide/route_guide_pb2_grpc.py

class RouteGuideStub(object):
  def __init__(self, channel):
	# 省略其它内容
    self.RouteChat = channel.stream_stream(
        '/routeguide.RouteGuide/RouteChat',
        request_serializer=route__guide__pb2.RouteNote.SerializeToString,
        response_deserializer=route__guide__pb2.RouteNote.FromString,
        )
```

调用 `RouteGuideStub.RouteChat()` 方法实际上是在调用 `channel.stream_stream()` 返回的 `grpc.StreamStreamMultiCallable` 对象.



## Channel



### 创建 Channel

```python
def insecure_channel(
    target: str, 
    options: List[tuple] = None, 
    compression: grpc.Compression = None,
):
    """Creates an insecure Channel to a server.

    The returned Channel is thread-safe.

    Args:
      target: The server address
      options: An optional list of key-value pairs (:term:`channel_arguments`
        in gRPC Core runtime) to configure the channel.
      compression: An optional value indicating the compression method to be
        used over the lifetime of the channel. This is an EXPERIMENTAL option.

    Returns:
      A Channel.
    """
    pass
```



###   接口定义 `grpc.Channel`

Channel 的接口 `grpc.Channel` 定义在 `src/python/grpcio/grpc/__init__.py` 中. 主体定义如下:

```python
class Channel(six.with_metaclass(abc.ABCMeta)):
    def subscribe(self, callback, try_to_connect=False):
        pass

    def unsubscribe(self, callback):
        pass

    def unary_unary(self, method, request_serializer=None, response_deserializer=None):
        pass

    def unary_stream(self, method, request_serializer=None, response_deserializer=None):
        pass

    def stream_unary(self, method, request_serializer=None, response_deserializer=None):
        pass

    def stream_stream(self, method, request_serializer=None, response_deserializer=None):
        pass

    def close(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
```



gRPC 中, Channel 本质上是 HTTP2 长连接的抽象, gRPC 请求应该尽可能地复用 Channel.



参考资料:

1. [Stack overflow: gRPC call, channel, connection and HTTP/2 lifecycle](https://stackoverflow.com/questions/63749113/grpc-call-channel-connection-and-http-2-lifecycle/63839453#63839453)
2. [Performance best practices with gRPC](https://docs.microsoft.com/en-us/aspnet/core/grpc/performance?view=aspnetcore-5.0)



### 接口实现 `grpc._channel.Channel`

模块 `src/python/grpcio/grpc/_channel.py` 给出了 `grpc.Channel` 接口的实现. 其初始化函数:

```python
class Channel(grpc.Channel):
    def __init__(
        self, 
        target: str, 
        options: List[tuple], 
        credentials: grpc.ChannelCredentials, 
        compression: grpc.Compression,
    ):
        python_options, core_options = _separate_channel_options(options)
        self._single_threaded_unary_stream = _DEFAULT_SINGLE_THREADED_UNARY_STREAM
        self._process_python_options(python_options)
        self._channel = cygrpc.Channel(
            _common.encode(target), _augment_options(core_options, compression),
            credentials)
        self._call_state = _ChannelCallState(self._channel)
        self._connectivity_state = _ChannelConnectivityState(self._channel)
        cygrpc.fork_register_channel(self)
```



由上述代码可知, `grpc._channel.Channel` 使用了 Cython 实现 `cygrpc.Channel`.



`cygrpc.Channel` 的初始化函数如下:

```python
cdef class Channel:

  def __cinit__(
      self, 
      bytes target, 
      object arguments,
      ChannelCredentials channel_credentials,
  ):
    arguments = () if arguments is None else tuple(arguments)
    fork_handlers_and_grpc_init()
    self._state = _ChannelState()
    self._state.c_call_completion_queue = (
        grpc_completion_queue_create_for_next(NULL))
    self._state.c_connectivity_completion_queue = (
        grpc_completion_queue_create_for_next(NULL))
    self._arguments = arguments
    cdef _ChannelArgs channel_args = _ChannelArgs(arguments)
    if channel_credentials is None:
      self._state.c_channel = grpc_insecure_channel_create(
          <char *>target, channel_args.c_args(), NULL)
    else:
      c_channel_credentials = channel_credentials.c()
      self._state.c_channel = grpc_secure_channel_create(
          c_channel_credentials, <char *>target, channel_args.c_args(), NULL)
      grpc_channel_credentials_release(c_channel_credentials)
```



`cygrpc.Channel` 会调用 C 函数 `grpc_insecure_channel_create()` 或 `grpc_secure_channel_create()` , 这两者都会返回 `grpc_channel` 对象.

`grpc_channel` 定义在 `src/core/lib/surface/channel.h` 文件中, 同时该文件还定义了一组操作 `grpc_channel` 的方法.



### Channel 的状态

`grpc.Channel` 有自己的状态, 状态使用 `grpc.ChannelConnectivity` 变量存储.

`grpc.ChannelConnectivity` 是一个 `enum.Enum` 枚举类型, 一共有五个不同的枚举变量:

```python
@enum.unique
class ChannelConnectivity(enum.Enum):
    IDLE = (_cygrpc.ConnectivityState.idle, 'idle')
    CONNECTING = (_cygrpc.ConnectivityState.connecting, 'connecting')
    READY = (_cygrpc.ConnectivityState.ready, 'ready')
    TRANSIENT_FAILURE = (_cygrpc.ConnectivityState.transient_failure,'transient failure')
    SHUTDOWN = (_cygrpc.ConnectivityState.shutdown, 'shutdown')
```



从上面的源码可以看到, 每个枚举变量都是一个 元组, 该元素表示了 `grpc.Channel` 的状态 与 `cygrpc.Channel` 的状态的对应关系. 后者的状态使用了枚举类 `cygrpc.ConnectivityState` (位于文件 `src/python/grpcio/grpc/_cython/_cygrpc/records.pyx.pxi` 中)来存储.



## Callable

前面讲到, 调用 Stub 的方法实际上是对相应的 `Callable` 类的调用.

gRPC 的四种接口分别对应了四种 `Callable`

* `grpc.UnaryUnaryMultiCallable`
* `grpc.UnaryStreamMultiCallable`
* `grpc.StreamUnaryMultiCallable`
* `grpc.StreamStreamMultiCallable`

上述 Callable 类在 `src/python/grpcio/grpc/__init__.py` 中给出了抽象接口,  在 `src/python/grpcio/grpc/_channel.py` 给出具体实现.



> 上述四种 Multi-Callable 中, `grpc.xxxUnaryyyy` 两个类表示返回的非流数据, 这两个类除了定义 `__call__()` 方法外, 还定义了
>
> * `future()` 实现异步调用
> * `with_call()` 实现同步调用
>
> 两个方法. 具体可以参考对应的接口定义和实现.



我们重点关注较为复杂的 `grpc.StreamStreamMultiCallable` 及其实现 `grpc._channel.StreamStreamMultiCallable`.



### 接口: `grpc.StreamStreamMultiCallable`

其调用参数为

```python
class StreamStreamMultiCallable(six.with_metaclass(abc.ABCMeta)):
    def __call__(
        self,
        request_iterator: Iterator,
        timeout: float = None,
        metadata: List[tuple] = None,
        credentials: grpc.CallCredentials = None,
        wait_for_ready: bool = None,
        compression: grpc.Compression = None,
    ):
        pass
```



### 实现: `grpc._channel._StreamStreamMultiCallable`

```python
class _StreamStreamMultiCallable(grpc.StreamStreamMultiCallable):

    # pylint: disable=too-many-arguments
    def __init__(
        self, 
        channel: grpc.Channel, 
        managed_call: cygrpc.IntegratedCall, 
        method: bytes, 
        request_serializer,
        response_deserializer,
    ):
        self._channel = channel
        self._managed_call = managed_call
        self._method = method
        self._request_serializer = request_serializer
        self._response_deserializer = response_deserializer
        self._context = cygrpc.build_census_context()

    def __call__(
        self,
        request_iterator: Iterator,
        timeout: float = None,
        metadata: List[tuple] = None,
        credentials: grpc.CallCredentials = None,
        wait_for_ready: bool = None,
        compression: grpc.Compression = None,
    ):
        deadline = _deadline(timeout)
        state = _RPCState(_STREAM_STREAM_INITIAL_DUE, None, None, None, None)
        initial_metadata_flags = _InitialMetadataFlags().with_wait_for_ready(wait_for_ready)
        augmented_metadata = _compression.augment_metadata(metadata, compression)
        operationses = (
            (
                cygrpc.SendInitialMetadataOperation(augmented_metadata,initial_metadata_flags),
                cygrpc.ReceiveStatusOnClientOperation(_EMPTY_FLAGS),
            ),
            (cygrpc.ReceiveInitialMetadataOperation(_EMPTY_FLAGS),),
        )
        event_handler = _event_handler(state, self._response_deserializer)
        call = self._managed_call(
            cygrpc.PropagationConstants.GRPC_PROPAGATE_DEFAULTS, 
            self._method,
            None, 
            _determine_deadline(deadline), 
            augmented_metadata,
            None if credentials is None else credentials._credentials,
            operationses, 
            event_handler, 
            self._context,
        )
        _consume_request_iterator(
            request_iterator, 
            state, 
            call,
            self._request_serializer, 
            event_handler,
        )
        return _MultiThreadedRendezvous(state, call, self._response_deserializer, deadline)
```



> **census**, 其中文释义为 `人口普查; (官方的)调查` 等, 来自 `self._context = cygrpc.build_census_context()`
>
> **rendezvous**, 其中文释义为 `约会, 聚会, 集合; 约会地点, 聚会场所` 等含义 , 来自 `_MultiThreadedRendezvous(state, call, self._response_deserializer, deadline)`



从上述代码可知, 输入流是通过 `grpc._channel._consume_request_iterator()` 方法实现的, 该方法内部为一个 `while True` 循环.



`grpc._channel.StreamStreamMultiCallable` 最终会由 `grpc._channel._MultiThreadedRendezvous` 类来实现.



继承关系图

```
grpc._channel._MultiThreadedRendezvous
	-> grpc._channel._Rendezvous
		-> grpc.RpcError -> Exception
		-> grpc.RpcContext -> abc.ABCMeta (metaclass)
	-> grpc.Call
		-> abc.ABCMeta (metaclass)
		-> grpc.RpcContext (metaclass) -> abc.ABCMeta (metaclass)
	-> grpc.Future
		-> abc.ABCMeta (metaclass)
```



其父类 `grpc._channel._Rendezvous` 实现了 `__iter__()` 和 `__next__()` 方法, 表明 `grpc._channel._MultiThreadedRendezvous` 既是迭代器, 也是可迭代对象.



## 客户端拦截器

gRPC 提供了四种客户端拦截器:

* `UnaryUnaryClientInterceptor`
* `UnaryStreamClientInterceptor`
* `StreamUnaryClientInterceptor`
* `StreamStreamClientInterceptor`

每个拦截器都定义了一个拦截器方法, 分别如下:

```python
class UnaryUnaryClientInterceptor(six.with_metaclass(abc.ABCMeta)):
    def intercept_unary_unary(self, continuation, client_call_details, request):
        pass
    
class UnaryStreamClientInterceptor(six.with_metaclass(abc.ABCMeta)):
    def intercept_unary_stream(self, continuation, client_call_details, request):
        pass
    
class StreamUnaryClientInterceptor(six.with_metaclass(abc.ABCMeta)):
    def intercept_stream_unary(self, continuation, client_call_details, request_iterator):
        pass
    
class StreamStreamClientInterceptor(six.with_metaclass(abc.ABCMeta)):
    def intercept_stream_stream(self, continuation, client_call_details, request_iterator):
        pass
```



这些接口拥有相似的函数签名:

* `continuation` 是一个函数, 该函数接收 `grpc.ClientCallDetails` 类型的参数, 也就是接口的下一个参数 `client_call_details`
* `client_call_details` 是一个 `grpc.ClientCallDetails` 类型参数
* `request` 和 `request_iterator` 是对应函数的请求参数



参数 `client_call_details` 记录了一个客户端请求的详细信息, 如:

* `method` 方法名
* `timeout` 超时时间
* `metadata` 请求元数据
* `credentials`  请求证书
* `wait_for_ready` 等待就绪标识, 具体参考 [术语 `wait_for_ready`](https://grpc.github.io/grpc/python/glossary.html#term-wait_for_ready)
* `compression` 客户端请求压缩方法
* 等等, 具体参考 `grpc.ClientCallDetails` 类实现



同样, gRPC 也没有提供内置的客户端拦截器实现. 客户端拦截器实现可以参考:

* `examples/python/interceptors/headers/generic_client_interceptor.py`
* `src/python/grpcio_tests/tests/unit/_interceptor_test.py`



`src/python/grpcio_tests/tests/unit/_interceptor_test.py` 文件中的 `_LoggingInterceptor` 给出了一个较好的拦截器实现示例.



## 客户端上下文

客户端上下文接口定义为 `grpc.Call` , 该接口继承了 `grpc.RpcContext`. 

`grpc.Call` 在 `grpc.RpcContext` 的基础上新增了 4 个接口.

```python
class Call(six.with_metaclass(abc.ABCMeta, RpcContext)):
    def initial_metadata(self):
        pass

    def trailing_metadata(self):
        pass

    def code(self):
        pass

    def details(self):
        pass
```



`grpc.Call` 及其子类 实际上就是一次 gRPC 请求的返回接口. `grpc.Channel` 的各个 Callable (如前面介绍的 `grpc._channel._StreamStreamMultiCallable` 等) 调用时的返回值都是 `grpc.Call` 的实现, 如:

* `grpc._channel._SingleThreadedRendezvous`
* `grpc._channel._MultiThreadedRendezvous`
* `grpc._channel._InactiveRpcError`
* `grpc._interceptor._FailureOutcome`
* `grpc._interceptor._UnaryOutcome`

* 等等

从这些实现来看, 它们都同时继承了 `grpc.Call` 和 `grpc.Future`.



客户端上下文中有一个重要的方法 `code()` , 客户端可以通过该方法获取服务端设置的状态码.  该状态码一般由服务端上下文 `grpc.ServicerContext` 的 `set_code()` 方法设置.



## 异步调用结果 Future

前面讲到, `grpc.Call` 代表 gRPC 请求的返回结果. 实现了 `grpc.Call` 接口的方法都实现了 `grpc.Future` 用于实现异步调用 gRPC.



`grpc.Future` 提供了如下方法:

```python
class Future(six.with_metaclass(abc.ABCMeta)):
    def cancel(self):
        pass

    def cancelled(self):
        pass

    def running(self):
        pass

    def done(self):
        pass

    def result(self, timeout=None):
        pass

    def exception(self, timeout=None):
        pass

    def traceback(self, timeout=None):
        pass

    def add_done_callback(self, fn):
        pass
```



除了前面提到的 `grpc.Call`  与 `grpc.Future`的共同子类外, `grpc.Future` 还有一个额外的子类 `grpc._utilities._ChannelReadyFuture`, 这个类由 `grpc.channel_ready_future()` 方法暴露给调用方使用.

