import os
import sys

if sys.version_info < (3, 11):
    raise RuntimeError("Python 3.11 or higher is required for the MCP plugin")
import re
import json
import struct
import threading
import http.server
from urllib.parse import urlparse
from typing import Any, Callable, get_type_hints, TypedDict, Optional, Annotated, TypeVar, Generic

class JSONRPCError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data

class RPCRegistry:
    def __init__(self):
        self.methods: dict[str, Callable] = {}
        self.unsafe: set[str] = set()

    def register(self, func: Callable) -> Callable:
        self.methods[func.__name__] = func
        return func

    def mark_unsafe(self, func: Callable) -> Callable:
        self.unsafe.add(func.__name__)
        return func

    def dispatch(self, method: str, params: Any) -> Any:
        if method not in self.methods:
            raise JSONRPCError(-32601, f"Method '{method}' not found")

        func = self.methods[method]
        hints = get_type_hints(func)

        # Remove return annotation if present
        hints.pop("return", None)

        if isinstance(params, list):
            if len(params) != len(hints):
                raise JSONRPCError(-32602, f"Invalid params: expected {len(hints)} arguments, got {len(params)}")

            # Validate and convert parameters
            converted_params = []
            for value, (param_name, expected_type) in zip(params, hints.items()):
                try:
                    if not isinstance(value, expected_type):
                        value = expected_type(value)
                    converted_params.append(value)
                except (ValueError, TypeError):
                    raise JSONRPCError(-32602, f"Invalid type for parameter '{param_name}': expected {expected_type.__name__}")

            return func(*converted_params)
        elif isinstance(params, dict):
            if set(params.keys()) != set(hints.keys()):
                raise JSONRPCError(-32602, f"Invalid params: expected {list(hints.keys())}")

            # Validate and convert parameters
            converted_params = {}
            for param_name, expected_type in hints.items():
                value = params.get(param_name)
                try:
                    if not isinstance(value, expected_type):
                        value = expected_type(value)
                    converted_params[param_name] = value
                except (ValueError, TypeError):
                    raise JSONRPCError(-32602, f"Invalid type for parameter '{param_name}': expected {expected_type.__name__}")

            return func(**converted_params)
        else:
            raise JSONRPCError(-32600, "Invalid Request: params must be array or object")

rpc_registry = RPCRegistry()

def jsonrpc(func: Callable) -> Callable:
    """Decorator to register a function as a JSON-RPC method"""
    global rpc_registry
    return rpc_registry.register(func)

def unsafe(func: Callable) -> Callable:
    """Decorator to register mark a function as unsafe"""
    return rpc_registry.mark_unsafe(func)

class JSONRPCRequestHandler(http.server.BaseHTTPRequestHandler):
    def send_jsonrpc_error(self, code: int, message: str, id: Any = None):
        response = {
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": message
            }
        }
        if id is not None:
            response["id"] = id
        response_body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response_body))
        self.end_headers()
        self.wfile.write(response_body)

    def do_POST(self):
        global rpc_registry

        parsed_path = urlparse(self.path)
        if parsed_path.path != "/mcp":
            self.send_jsonrpc_error(-32098, "Invalid endpoint", None)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self.send_jsonrpc_error(-32700, "Parse error: missing request body", None)
            return

        request_body = self.rfile.read(content_length)
        try:
            request = json.loads(request_body)
        except json.JSONDecodeError:
            self.send_jsonrpc_error(-32700, "Parse error: invalid JSON", None)
            return

        # Prepare the response
        response = {
            "jsonrpc": "2.0"
        }
        if request.get("id") is not None:
            response["id"] = request.get("id")

        try:
            # Basic JSON-RPC validation
            if not isinstance(request, dict):
                raise JSONRPCError(-32600, "Invalid Request")
            if request.get("jsonrpc") != "2.0":
                raise JSONRPCError(-32600, "Invalid JSON-RPC version")
            if "method" not in request:
                raise JSONRPCError(-32600, "Method not specified")

            # Dispatch the method
            result = rpc_registry.dispatch(request["method"], request.get("params", []))
            response["result"] = result

        except JSONRPCError as e:
            response["error"] = {
                "code": e.code,
                "message": e.message
            }
            if e.data is not None:
                response["error"]["data"] = e.data
        except IDAError as e:
            response["error"] = {
                "code": -32000,
                "message": e.message,
            }
        except Exception as e:
            traceback.print_exc()
            response["error"] = {
                "code": -32603,
                "message": "Internal error (please report a bug)",
                "data": traceback.format_exc(),
            }

        try:
            response_body = json.dumps(response).encode("utf-8")
        except Exception as e:
            traceback.print_exc()
            response_body = json.dumps({
                "error": {
                    "code": -32603,
                    "message": "Internal error (please report a bug)",
                    "data": traceback.format_exc(),
                }
            }).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response_body))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format, *args):
        # Suppress logging
        pass

class MCPHTTPServer(http.server.HTTPServer):
    allow_reuse_address = False

class Server:
    HOST = "localhost"
    PORT = 13337

    def __init__(self):
        self.server = None
        self.server_thread = None
        self.running = False

    def start(self):
        if self.running:
            print("[MCP] Server is already running")
            return

        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.running = True
        self.server_thread.start()

    def stop(self):
        if not self.running:
            return

        self.running = False
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread:
            self.server_thread.join()
            self.server = None
        print("[MCP] Server stopped")

    def _run_server(self):
        try:
            # Create server in the thread to handle binding
            self.server = MCPHTTPServer((Server.HOST, Server.PORT), JSONRPCRequestHandler)
            print(f"[MCP] Server started at http://{Server.HOST}:{Server.PORT}")
            self.server.serve_forever()
        except OSError as e:
            if e.errno == 98 or e.errno == 10048:  # Port already in use (Linux/Windows)
                print("[MCP] Error: Port 13337 is already in use")
            else:
                print(f"[MCP] Server error: {e}")
            self.running = False
        except Exception as e:
            print(f"[MCP] Server error: {e}")
        finally:
            self.running = False

# A module that helps with writing thread safe ida code.
# Based on:
# https://web.archive.org/web/20160305190440/http://www.williballenthin.com/blog/2015/09/04/idapython-synchronization-decorator/
import logging
import queue
import traceback
import functools

import ida_hexrays
import ida_kernwin
import ida_funcs
import ida_gdl
import ida_lines
import ida_idaapi
import idc
import idaapi
import idautils
import ida_nalt
import ida_bytes
import ida_typeinf
import ida_ua
import ida_xref
import ida_entry
import idautils
import ida_idd
import ida_dbg
import ida_name
import ida_ida

class IDAError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

    @property
    def message(self) -> str:
        return self.args[0]

class IDASyncError(Exception):
    pass

# Important note: Always make sure the return value from your function f is a
# copy of the data you have gotten from IDA, and not the original data.
#
# Example:
# --------
#
# Do this:
#
#   @idaread
#   def ts_Functions():
#       return list(idautils.Functions())
#
# Don't do this:
#
#   @idaread
#   def ts_Functions():
#       return idautils.Functions()
#

logger = logging.getLogger(__name__)

# Enum for safety modes. Higher means safer:
class IDASafety:
    ida_kernwin.MFF_READ
    SAFE_NONE = ida_kernwin.MFF_FAST
    SAFE_READ = ida_kernwin.MFF_READ
    SAFE_WRITE = ida_kernwin.MFF_WRITE

call_stack = queue.LifoQueue()

def sync_wrapper(ff, safety_mode: IDASafety):
    """
    Call a function ff with a specific IDA safety_mode.
    """
    #logger.debug('sync_wrapper: {}, {}'.format(ff.__name__, safety_mode))

    if safety_mode not in [IDASafety.SAFE_READ, IDASafety.SAFE_WRITE]:
        error_str = 'Invalid safety mode {} over function {}'\
                .format(safety_mode, ff.__name__)
        logger.error(error_str)
        raise IDASyncError(error_str)

    # No safety level is set up:
    res_container = queue.Queue()

    def runned():
        #logger.debug('Inside runned')

        # Make sure that we are not already inside a sync_wrapper:
        if not call_stack.empty():
            last_func_name = call_stack.get()
            error_str = ('Call stack is not empty while calling the '
                'function {} from {}').format(ff.__name__, last_func_name)
            #logger.error(error_str)
            raise IDASyncError(error_str)

        call_stack.put((ff.__name__))
        try:
            res_container.put(ff())
        except Exception as x:
            res_container.put(x)
        finally:
            call_stack.get()
            #logger.debug('Finished runned')

    ret_val = idaapi.execute_sync(runned, safety_mode)
    res = res_container.get()
    if isinstance(res, Exception):
        raise res
    return res

def idawrite(f):
    """
    decorator for marking a function as modifying the IDB.
    schedules a request to be made in the main IDA loop to avoid IDB corruption.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        ff = functools.partial(f, *args, **kwargs)
        ff.__name__ = f.__name__
        return sync_wrapper(ff, idaapi.MFF_WRITE)
    return wrapper

def idaread(f):
    """
    decorator for marking a function as reading from the IDB.
    schedules a request to be made in the main IDA loop to avoid
      inconsistent results.
    MFF_READ constant via: http://www.openrce.org/forums/posts/1827
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        ff = functools.partial(f, *args, **kwargs)
        ff.__name__ = f.__name__
        return sync_wrapper(ff, idaapi.MFF_READ)
    return wrapper

def is_window_active():
    """Returns whether IDA is currently active"""
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        return False

    app = QApplication.instance()
    if app is None:
        return False

    for widget in app.topLevelWidgets():
        if widget.isActiveWindow():
            return True
    return False

class Metadata(TypedDict):
    path: str
    module: str
    base: str
    size: str
    md5: str
    sha256: str
    crc32: str
    filesize: str

def get_image_size():
    try:
        # https://www.hex-rays.com/products/ida/support/sdkdoc/structidainfo.html
        info = idaapi.get_inf_structure()
        omin_ea = info.omin_ea
        omax_ea = info.omax_ea
    except AttributeError:
        import ida_ida
        omin_ea = ida_ida.inf_get_omin_ea()
        omax_ea = ida_ida.inf_get_omax_ea()
    # Bad heuristic for image size (bad if the relocations are the last section)
    image_size = omax_ea - omin_ea
    # Try to extract it from the PE header
    header = idautils.peutils_t().header()
    if header and header[:4] == b"PE\0\0":
        image_size = struct.unpack("<I", header[0x50:0x54])[0]
    return image_size

@jsonrpc
@idaread
def get_metadata() -> Metadata:
    """Get metadata about the current IDB"""
    # Fat Mach-O binaries can return a None hash:
    # https://github.com/mrexodia/ida-pro-mcp/issues/26
    def hash(f):
        try:
            return f().hex()
        except:
            return None
    return {
        "path": idaapi.get_input_file_path(),
        "module": idaapi.get_root_filename(),
        "base": hex(idaapi.get_imagebase()),
        "size": hex(get_image_size()),
        "md5": hash(ida_nalt.retrieve_input_file_md5),
        "sha256": hash(ida_nalt.retrieve_input_file_sha256),
        "crc32": hex(ida_nalt.retrieve_input_file_crc32()),
        "filesize": hex(ida_nalt.retrieve_input_file_size()),
    }

def get_prototype(fn: ida_funcs.func_t) -> Optional[str]:
    try:
        prototype: ida_typeinf.tinfo_t = fn.get_prototype()
        if prototype is not None:
            return str(prototype)
        else:
            return None
    except AttributeError:
        try:
            return idc.get_type(fn.start_ea)
        except:
            tif = ida_typeinf.tinfo_t()
            if ida_nalt.get_tinfo(tif, fn.start_ea):
                return str(tif)
            return None
    except Exception as e:
        print(f"Error getting function prototype: {e}")
        return None

class Function(TypedDict):
    address: str
    name: str
    size: str

def parse_address(address: str) -> int:
    try:
        return int(address, 0)
    except ValueError:
        for ch in address:
            if ch not in "0123456789abcdefABCDEF":
                raise IDAError(f"Failed to parse address: {address}")
        raise IDAError(f"Failed to parse address (missing 0x prefix): {address}")

def get_function(address: int, *, raise_error=True) -> Function:
    fn = idaapi.get_func(address)
    if fn is None:
        if raise_error:
            raise IDAError(f"No function found at address {hex(address)}")
        return None

    try:
        name = fn.get_name()
    except AttributeError:
        name = ida_funcs.get_func_name(fn.start_ea)
    return {
        "address": hex(fn.start_ea),
        "name": name,
        "size": hex(fn.end_ea - fn.start_ea),
    }

DEMANGLED_TO_EA = {}

def create_demangled_to_ea_map():
    for ea in idautils.Functions():
        # Get the function name and demangle it
        # MNG_NODEFINIT inhibits everything except the main name
        # where default demangling adds the function signature
        # and decorators (if any)
        demangled = idaapi.demangle_name(
            idc.get_name(ea, 0), idaapi.MNG_NODEFINIT)
        if demangled:
            DEMANGLED_TO_EA[demangled] = ea

@jsonrpc
@idaread
def get_function_by_name(
    name: Annotated[str, "Name of the function to get"]
) -> Function:
    """Get a function by its name"""
    function_address = idaapi.get_name_ea(idaapi.BADADDR, name)
    if function_address == idaapi.BADADDR:
        # If map has not been created yet, create it
        if len(DEMANGLED_TO_EA) == 0:
            create_demangled_to_ea_map()
        # Try to find the function in the map, else raise an error
        if name in DEMANGLED_TO_EA:
            function_address = DEMANGLED_TO_EA[name]
        else:
            raise IDAError(f"No function found with name {name}")
    return get_function(function_address)

@jsonrpc
@idaread
def get_function_by_address(
    address: Annotated[str, "Address of the function to get"],
) -> Function:
    """Get a function by its address"""
    return get_function(parse_address(address))

@jsonrpc
@idaread
def get_current_address() -> str:
    """Get the address currently selected by the user"""
    return hex(idaapi.get_screen_ea())

@jsonrpc
@idaread
def get_current_function() -> Optional[Function]:
    """Get the function currently selected by the user"""
    return get_function(idaapi.get_screen_ea())

class ConvertedNumber(TypedDict):
    decimal: str
    hexadecimal: str
    bytes: str
    ascii: Optional[str]
    binary: str

@jsonrpc
def convert_number(
    text: Annotated[str, "Textual representation of the number to convert"],
    size: Annotated[Optional[int], "Size of the variable in bytes"],
) -> ConvertedNumber:
    """Convert a number (decimal, hexadecimal) to different representations"""
    try:
        value = int(text, 0)
    except ValueError:
        raise IDAError(f"Invalid number: {text}")

    # Estimate the size of the number
    if not size:
        size = 0
        n = abs(value)
        while n:
            size += 1
            n >>= 1
        size += 7
        size //= 8

    # Convert the number to bytes
    try:
        bytes = value.to_bytes(size, "little", signed=True)
    except OverflowError:
        raise IDAError(f"Number {text} is too big for {size} bytes")

    # Convert the bytes to ASCII
    ascii = ""
    for byte in bytes.rstrip(b"\x00"):
        if byte >= 32 and byte <= 126:
            ascii += chr(byte)
        else:
            ascii = None
            break

    return {
        "decimal": str(value),
        "hexadecimal": hex(value),
        "bytes": bytes.hex(" "),
        "ascii": ascii,
        "binary": bin(value),
    }

T = TypeVar("T")

class Page(TypedDict, Generic[T]):
    data: list[T]
    next_offset: Optional[int]

def paginate(data: list[T], offset: int, count: int) -> Page[T]:
    if count == 0:
        count = len(data)
    next_offset = offset + count
    if next_offset >= len(data):
        next_offset = None
    return {
        "data": data[offset:offset + count],
        "next_offset": next_offset,
    }

def pattern_filter(data: list[T], pattern: str, key: str) -> list[T]:
    if not pattern:
        return data

    # TODO: implement /regex/ matching

    def matches(item: T) -> bool:
        return pattern.lower() in item[key].lower()
    return list(filter(matches, data))

@jsonrpc
@idaread
def list_functions(
    offset: Annotated[int, "Offset to start listing from (start at 0)"],
    count: Annotated[int, "Number of functions to list (100 is a good default, 0 means remainder)"],
) -> Page[Function]:
    """List all functions in the database (paginated)"""
    functions = [get_function(address) for address in idautils.Functions()]
    return paginate(functions, offset, count)

class Global(TypedDict):
    address: str
    name: str

@jsonrpc
@idaread
def list_globals_filter(
    offset: Annotated[int, "Offset to start listing from (start at 0)"],
    count: Annotated[int, "Number of globals to list (100 is a good default, 0 means remainder)"],
    filter: Annotated[str, "Filter to apply to the list (required parameter, empty string for no filter). Case-insensitive contains or /regex/ syntax"],
) -> Page[Global]:
    """List matching globals in the database (paginated, filtered)"""
    globals = []
    for addr, name in idautils.Names():
        # Skip functions
        if not idaapi.get_func(addr):
            globals.append({
                "address": hex(addr),
                "name": name,
            })
    globals = pattern_filter(globals, filter, "name")
    return paginate(globals, offset, count)

@jsonrpc
def list_globals(
    offset: Annotated[int, "Offset to start listing from (start at 0)"],
    count: Annotated[int, "Number of globals to list (100 is a good default, 0 means remainder)"],
) -> Page[Global]:
    """List all globals in the database (paginated)"""
    return list_globals_filter(offset, count, "")

class String(TypedDict):
    address: str
    length: int
    string: str

@jsonrpc
@idaread
def list_strings_filter(
    offset: Annotated[int, "Offset to start listing from (start at 0)"],
    count: Annotated[int, "Number of strings to list (100 is a good default, 0 means remainder)"],
    filter: Annotated[str, "Filter to apply to the list (required parameter, empty string for no filter). Case-insensitive contains or /regex/ syntax"],
) -> Page[String]:
    """List matching strings in the database (paginated, filtered)"""
    strings = []
    for item in idautils.Strings():
        try:
            string = str(item)
            if string:
                strings.append({
                    "address": hex(item.ea),
                    "length": item.length,
                    "string": string,
                })
        except:
            continue
    strings = pattern_filter(strings, filter, "string")
    return paginate(strings, offset, count)

@jsonrpc
def list_strings(
    offset: Annotated[int, "Offset to start listing from (start at 0)"],
    count: Annotated[int, "Number of strings to list (100 is a good default, 0 means remainder)"],
) -> Page[String]:
    """List all strings in the database (paginated)"""
    return list_strings_filter(offset, count, "")

class ImportedFunction(TypedDict):
    name: str
    address: str # The address in the IAT
    # ordinal: Optional[int] # Could add if needed

class ImportedModule(TypedDict):
    module_name: str
    functions: list[ImportedFunction]

@jsonrpc
@idaread
def list_imports() -> list[ImportedModule]:
    """List all imported modules and their functions."""
    results = []
    num_modules = ida_nalt.get_import_module_qty()

    for i in range(num_modules):
        module_name = ida_nalt.get_import_module_name(i)
        if not module_name:
            continue

        imported_functions: list[ImportedFunction] = []
        
        def imp_cb(ea, name, ord):
            # True -> Continue enumeration
            # False -> Stop enumeration
            if not name: # Sometimes names are not available, only ordinals
                name = f"ord_{ord}"
            imported_functions.append({
                "name": name,
                "address": hex(ea),
                # "ordinal": ord
            })
            return True

        ida_nalt.enum_import_names(i, imp_cb)
        results.append({
            "module_name": module_name,
            "functions": imported_functions
        })
        
    return results

class FoundBytes(TypedDict):
    address: str
    # context: Optional[str] # Could add context later if needed

@jsonrpc
@idaread
def find_bytes(
    bytes_hex_string: Annotated[str, "Hex string of bytes to search for (e.g., '43100000')"],
    search_start_address: Annotated[Optional[str], "Optional start address for search (hex)"] = None,
    search_end_address: Annotated[Optional[str], "Optional end address for search (hex, exclusive)"] = None
) -> list[FoundBytes]:
    """Search for a sequence of bytes in the database."""
    try:
        search_bytes = bytes.fromhex(bytes_hex_string)
    except ValueError:
        raise IDAError(f"Invalid hex string for bytes: {bytes_hex_string}")

    if not search_bytes:
        raise IDAError("Byte string to search cannot be empty.")

    start_ea = ida_ida.inf_get_min_ea() # Corrected API
    if search_start_address:
        start_ea = parse_address(search_start_address)

    end_ea = ida_ida.inf_get_max_ea() # Corrected API
    if search_end_address:
        end_ea = parse_address(search_end_address)
    
    results: list[FoundBytes] = []
    current_ea = start_ea
    
    # Using a simpler loop for now, ida_bytes.find_binary can be complex with its flags.
    # We'll iterate through segments or use a chunked read.
    # For simplicity and to avoid issues with find_binary over very large selections or specific IDA versions,
    # let's iterate segments. This is less efficient than a direct find_binary over all memory
    # but more robust for a plugin. A chunked read approach is better.

    chunk_size = 1024 * 1024 # 1MB chunks
    ea = start_ea
    while ea < end_ea:
        # Check for user cancellation periodically
        if ida_kernwin.user_cancelled():
            print("[MCP] Byte search cancelled by user.")
            break

        read_length = min(chunk_size, end_ea - ea)
        if read_length <= 0:
            break
        
        data_chunk = ida_bytes.get_bytes(ea, read_length)
        
        if data_chunk is None: # Reached unreadable memory or end
            # This might happen if a segment is smaller than expected or unreadable
            # Advance ea by a smaller step or segment step if this becomes an issue
            break
            
        offset_in_chunk = 0
        while True:
            found_offset = data_chunk.find(search_bytes, offset_in_chunk)
            if found_offset == -1:
                break
            
            actual_address = ea + found_offset
            results.append({"address": hex(actual_address)})
            
            if len(results) > 200: # Safety break for too many results
                 raise IDAError(f"Too many results (>200) for byte search, please refine search or range.")
            
            offset_in_chunk = found_offset + 1 # Continue search in the rest of the chunk
        
        ea += read_length
            
    return results

class FoundImmediate(TypedDict):
    address: str
    instruction: str
    # operand_index: int # Could add if needed

@jsonrpc
@idaread
def find_immediate(
    immediate_value: Annotated[str, "Immediate value to search for (hex or decimal)"],
    search_start_address: Annotated[Optional[str], "Optional start address for search (hex)"] = None,
    search_end_address: Annotated[Optional[str], "Optional end address for search (hex, exclusive)"] = None
) -> list[FoundImmediate]:
    """Search for an immediate value in code sections."""
    try:
        value_to_find = parse_address(immediate_value) # Use existing parse_address
    except IDAError: # If parse_address fails (e.g. not hex and not valid int string)
        raise IDAError(f"Invalid immediate value: {immediate_value}. Must be hex (0x...) or decimal.")

    start_ea = ida_ida.inf_get_min_ea() # Corrected API
    if search_start_address:
        start_ea = parse_address(search_start_address)

    end_ea = ida_ida.inf_get_max_ea() # Corrected API
    if search_end_address:
        end_ea = parse_address(search_end_address)

    results: list[FoundImmediate] = []

    for func_ea in idautils.Functions(start_ea, end_ea):
        func = ida_funcs.get_func(func_ea)
        if not func:
            continue

        current_instr_ea = max(func.start_ea, start_ea)
        func_search_end_ea = min(func.end_ea, end_ea) # Ensure search within overall bounds

        while current_instr_ea < func_search_end_ea and current_instr_ea != ida_idaapi.BADADDR:
            if ida_kernwin.user_cancelled():
                print("[MCP] Immediate search cancelled by user.")
                return results

            instr_text = idc.GetDisasm(current_instr_ea) # Corrected: idc.get_disasm to idc.GetDisasm
            
            for i in range(6):
                op_type = idc.get_operand_type(current_instr_ea, i)
                
                if op_type == ida_ua.o_imm: # Immediate value
                    op_value = idc.get_operand_value(current_instr_ea, i)
                    if op_value == value_to_find:
                        results.append({
                            "address": hex(current_instr_ea),
                            "instruction": instr_text
                        })
                        if len(results) > 200:
                            raise IDAError("Too many results (>200) for immediate search, please refine.")
                        break
                elif op_type == ida_ua.o_displ: # Displacement [reg + immediate_offset]
                    op_value = idc.get_operand_value(current_instr_ea, i)
                    if op_value == value_to_find:
                        results.append({
                            "address": hex(current_instr_ea),
                            "instruction": instr_text
                        })
                        if len(results) > 200:
                            raise IDAError("Too many results (>200) for immediate search, please refine.")
                        break
                elif op_type == ida_ua.o_near or op_type == ida_ua.o_far: # Address for call/jmp
                    op_value = idc.get_operand_value(current_instr_ea, i)
                    if op_value == value_to_find:
                        results.append({
                            "address": hex(current_instr_ea),
                            "instruction": instr_text
                        })
                        if len(results) > 200:
                            raise IDAError("Too many results (>200) for immediate search, please refine.")
                        break
                
            next_instr_ea = ida_bytes.next_head(current_instr_ea, func_search_end_ea)
            if next_instr_ea <= current_instr_ea :
                break
            current_instr_ea = next_instr_ea
            
    return results

def decompile_checked(address: int) -> ida_hexrays.cfunc_t:
    if not ida_hexrays.init_hexrays_plugin():
        raise IDAError("Hex-Rays decompiler is not available")
    error = ida_hexrays.hexrays_failure_t()
    cfunc: ida_hexrays.cfunc_t = ida_hexrays.decompile_func(address, error, ida_hexrays.DECOMP_WARNINGS)
    if not cfunc:
        message = f"Decompilation failed at {hex(address)}"
        if error.str:
            message += f": {error.str}"
        if error.errea != idaapi.BADADDR:
            message += f" (address: {hex(error.errea)})"
        raise IDAError(message)
    return cfunc

@jsonrpc
@idaread
def decompile_function(
    address: Annotated[str, "Address of the function to decompile"],
) -> str:
    """Decompile a function at the given address"""
    address = parse_address(address)
    cfunc = decompile_checked(address)
    if is_window_active():
        ida_hexrays.open_pseudocode(address, ida_hexrays.OPF_REUSE)
    sv = cfunc.get_pseudocode()
    pseudocode = ""
    for i, sl in enumerate(sv):
        sl: ida_kernwin.simpleline_t
        item = ida_hexrays.ctree_item_t()
        addr = None if i > 0 else cfunc.entry_ea
        if cfunc.get_line_item(sl.line, 0, False, None, item, None):
            ds = item.dstr().split(": ")
            if len(ds) == 2:
                try:
                    addr = int(ds[0], 16)
                except ValueError:
                    pass
        line = ida_lines.tag_remove(sl.line)
        if len(pseudocode) > 0:
            pseudocode += "\n"
        if not addr:
            pseudocode += f"/* line: {i} */ {line}"
        else:
            pseudocode += f"/* line: {i}, address: {hex(addr)} */ {line}"

    return pseudocode

@jsonrpc
@idaread
def disassemble_function(
    start_address: Annotated[str, "Address of the function to disassemble"],
) -> str:
    """Get assembly code (address: instruction; comment) for a function"""
    start = parse_address(start_address)
    func = idaapi.get_func(start)
    if not func:
        raise IDAError(f"No function found containing address {start_address}")
    if is_window_active():
        ida_kernwin.jumpto(start)

    # TODO: add labels and limit the maximum number of instructions
    disassembly = ""
    for address in ida_funcs.func_item_iterator_t(func):
        if len(disassembly) > 0:
            disassembly += "\n"
        disassembly += f"{hex(address)}: "
        disassembly += idaapi.generate_disasm_line(address, idaapi.GENDSM_REMOVE_TAGS)
        comment = idaapi.get_cmt(address, False)
        if not comment:
            comment = idaapi.get_cmt(address, True)
        if comment:
            disassembly += f"; {comment}"
    return disassembly

class Xref(TypedDict):
    address: str
    type: str
    function: Optional[Function]
class ModuleInfo(TypedDict):
    name: str       # Full path of the module
    base: str       # Hex base address
    size: int       # Size in bytes
    rebase_to: str  # Hex rebase_to address (if IDA rebased it)
    name: str       # Full path of the module
    base: str       # Hex base address
    size: int       # Size in bytes
    rebase_to: str  # Hex rebase_to address (if IDA rebased it)

@jsonrpc
@idaread
def get_xrefs_to(
    address: Annotated[str, "Address to get cross references to"],
) -> list[Xref]:
    """Get all cross references to the given address"""
    xrefs = []
    xref: ida_xref.xrefblk_t
    for xref in idautils.XrefsTo(parse_address(address)):
        xrefs.append({
            "address": hex(xref.frm),
            "type": "code" if xref.iscode else "data",
            "function": get_function(xref.frm, raise_error=False),
        })
    return xrefs

@jsonrpc
@idaread
def get_xrefs_to_field(
    struct_name: Annotated[str, "Name of the struct (type) containing the field"],
    field_name: Annotated[str, "Name of the field (member) to get xrefs to"],
) -> list[Xref]:
    """Get all cross references to a named struct field (member)"""

    # Get the type library
    til = ida_typeinf.get_idati()
    if not til:
        raise IDAError("Failed to retrieve type library.")

    # Get the structure type info
    tif = ida_typeinf.tinfo_t()
    if not tif.get_named_type(til, struct_name, ida_typeinf.BTF_STRUCT, True, False):
        print(f"Structure '{struct_name}' not found.")
        return []

    # Get The field index
    idx = ida_typeinf.get_udm_by_fullname(None, struct_name + '.' + field_name)
    if idx == -1:
        print(f"Field '{field_name}' not found in structure '{struct_name}'.")
        return []

    # Get the type identifier
    tid = tif.get_udm_tid(idx)
    if tid == ida_idaapi.BADADDR:
        raise IDAError(f"Unable to get tid for structure '{struct_name}' and field '{field_name}'.")

    # Get xrefs to the tid
    xrefs = []
    xref: ida_xref.xrefblk_t
    for xref in idautils.XrefsTo(tid):
        xrefs.append({
            "address": hex(xref.frm),
            "type": "code" if xref.iscode else "data",
            "function": get_function(xref.frm, raise_error=False),
        })
    return xrefs

@jsonrpc
@idaread
def get_entry_points() -> list[Function]:
    """Get all entry points in the database"""
    result = []
    for i in range(ida_entry.get_entry_qty()):
        ordinal = ida_entry.get_entry_ordinal(i)
        address = ida_entry.get_entry(ordinal)
        func = get_function(address, raise_error=False)
        if func is not None:
            result.append(func)
    return result

@jsonrpc
@idawrite
def set_comment(
    address: Annotated[str, "Address in the function to set the comment for"],
    comment: Annotated[str, "Comment text"],
):
    """Set a comment for a given address in the function disassembly and pseudocode"""
    address = parse_address(address)

    if not idaapi.set_cmt(address, comment, False):
        raise IDAError(f"Failed to set disassembly comment at {hex(address)}")

    # Reference: https://cyber.wtf/2019/03/22/using-ida-python-to-analyze-trickbot/
    # Check if the address corresponds to a line
    cfunc = decompile_checked(address)

    # Special case for function entry comments
    if address == cfunc.entry_ea:
        idc.set_func_cmt(address, comment, True)
        cfunc.refresh_func_ctext()
        return

    eamap = cfunc.get_eamap()
    if address not in eamap:
        print(f"Failed to set decompiler comment at {hex(address)}")
        return
    nearest_ea = eamap[address][0].ea

    # Remove existing orphan comments
    if cfunc.has_orphan_cmts():
        cfunc.del_orphan_cmts()
        cfunc.save_user_cmts()

    # Set the comment by trying all possible item types
    tl = idaapi.treeloc_t()
    tl.ea = nearest_ea
    for itp in range(idaapi.ITP_SEMI, idaapi.ITP_COLON):
        tl.itp = itp
        cfunc.set_user_cmt(tl, comment)
        cfunc.save_user_cmts()
        cfunc.refresh_func_ctext()
        if not cfunc.has_orphan_cmts():
            return
        cfunc.del_orphan_cmts()
        cfunc.save_user_cmts()
    print(f"Failed to set decompiler comment at {hex(address)}")

def refresh_decompiler_widget():
    widget = ida_kernwin.get_current_widget()
    if widget is not None:
        vu = ida_hexrays.get_widget_vdui(widget)
        if vu is not None:
            vu.refresh_ctext()

def refresh_decompiler_ctext(function_address: int):
    error = ida_hexrays.hexrays_failure_t()
    cfunc: ida_hexrays.cfunc_t = ida_hexrays.decompile_func(function_address, error, ida_hexrays.DECOMP_WARNINGS)
    if cfunc:
        cfunc.refresh_func_ctext()

@jsonrpc
@idawrite
def rename_local_variable(
    function_address: Annotated[str, "Address of the function containing the variable"],
    old_name: Annotated[str, "Current name of the variable"],
    new_name: Annotated[str, "New name for the variable (empty for a default name)"],
):
    """Rename a local variable in a function"""
    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")
    if not ida_hexrays.rename_lvar(func.start_ea, old_name, new_name):
        raise IDAError(f"Failed to rename local variable {old_name} in function {hex(func.start_ea)}")
    refresh_decompiler_ctext(func.start_ea)

@jsonrpc
@idawrite
def rename_global_variable(
    old_name: Annotated[str, "Current name of the global variable"],
    new_name: Annotated[str, "New name for the global variable (empty for a default name)"],
):
    """Rename a global variable"""
    ea = idaapi.get_name_ea(idaapi.BADADDR, old_name)
    if not idaapi.set_name(ea, new_name):
        raise IDAError(f"Failed to rename global variable {old_name} to {new_name}")
    refresh_decompiler_ctext(ea)

@jsonrpc
@idawrite
def set_global_variable_type(
    variable_name: Annotated[str, "Name of the global variable"],
    new_type: Annotated[str, "New type for the variable"],
):
    """Set a global variable's type"""
    ea = idaapi.get_name_ea(idaapi.BADADDR, variable_name)
    tif = ida_typeinf.tinfo_t(new_type, None, ida_typeinf.PT_SIL)
    if not tif:
        raise IDAError(f"Parsed declaration is not a variable type")
    if not ida_typeinf.apply_tinfo(ea, tif, ida_typeinf.PT_SIL):
        raise IDAError(f"Failed to apply type")

@jsonrpc
@idawrite
def rename_function(
    function_address: Annotated[str, "Address of the function to rename"],
    new_name: Annotated[str, "New name for the function (empty for a default name)"],
):
    """Rename a function"""
    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")
    if not idaapi.set_name(func.start_ea, new_name):
        raise IDAError(f"Failed to rename function {hex(func.start_ea)} to {new_name}")
    refresh_decompiler_ctext(func.start_ea)

@jsonrpc
@idawrite
def set_function_prototype(
    function_address: Annotated[str, "Address of the function"],
    prototype: Annotated[str, "New function prototype"],
) -> str:
    """Set a function's prototype"""
    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")
    try:
        tif = ida_typeinf.tinfo_t(prototype, None, ida_typeinf.PT_SIL)
        if not tif.is_func():
            raise IDAError(f"Parsed declaration is not a function type")
        if not ida_typeinf.apply_tinfo(func.start_ea, tif, ida_typeinf.PT_SIL):
            raise IDAError(f"Failed to apply type")
        refresh_decompiler_ctext(func.start_ea)
    except Exception as e:
        raise IDAError(f"Failed to parse prototype string: {prototype}")

class my_modifier_t(ida_hexrays.user_lvar_modifier_t):
    def __init__(self, var_name: str, new_type: ida_typeinf.tinfo_t):
        ida_hexrays.user_lvar_modifier_t.__init__(self)
        self.var_name = var_name
        self.new_type = new_type

    def modify_lvars(self, lvars):
        for lvar_saved in lvars.lvvec:
            lvar_saved: ida_hexrays.lvar_saved_info_t
            if lvar_saved.name == self.var_name:
                lvar_saved.type = self.new_type
                return True
        return False

# NOTE: This is extremely hacky, but necessary to get errors out of IDA
def parse_decls_ctypes(decls: str, hti_flags: int) -> tuple[int, str]:
    if sys.platform == "win32":
        import ctypes

        assert isinstance(decls, str), "decls must be a string"
        assert isinstance(hti_flags, int), "hti_flags must be an int"
        c_decls = decls.encode("utf-8")
        c_til = None
        ida_dll = ctypes.CDLL("ida")
        ida_dll.parse_decls.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        ida_dll.parse_decls.restype = ctypes.c_int

        messages = []

        @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p)
        def magic_printer(fmt: bytes, arg1: bytes):
            if fmt.count(b"%") == 1 and b"%s" in fmt:
                formatted = fmt.replace(b"%s", arg1)
                messages.append(formatted.decode("utf-8"))
                return len(formatted) + 1
            else:
                messages.append(f"unsupported magic_printer fmt: {repr(fmt)}")
                return 0

        errors = ida_dll.parse_decls(c_til, c_decls, magic_printer, hti_flags)
    else:
        # NOTE: The approach above could also work on other platforms, but it's
        # not been tested and there are differences in the vararg ABIs.
        errors = ida_typeinf.parse_decls(None, decls, False, hti_flags)
        messages = []
    return errors, messages

@jsonrpc
@idawrite
def declare_c_type(
    c_declaration: Annotated[str, "C declaration of the type. Examples include: typedef int foo_t; struct bar { int a; bool b; };"],
):
    """Create or update a local type from a C declaration"""
    # PT_SIL: Suppress warning dialogs (although it seems unnecessary here)
    # PT_EMPTY: Allow empty types (also unnecessary?)
    # PT_TYP: Print back status messages with struct tags
    flags = ida_typeinf.PT_SIL | ida_typeinf.PT_EMPTY | ida_typeinf.PT_TYP
    errors, messages = parse_decls_ctypes(c_declaration, flags)

    pretty_messages = "\n".join(messages)
    if errors > 0:
        raise IDAError(f"Failed to parse type:\n{c_declaration}\n\nErrors:\n{pretty_messages}")
    return f"success\n\nInfo:\n{pretty_messages}"

@jsonrpc
@idawrite
def set_local_variable_type(
    function_address: Annotated[str, "Address of the function containing the variable"],
    variable_name: Annotated[str, "Name of the variable"],
    new_type: Annotated[str, "New type for the variable"],
):
    """Set a local variable's type"""
    try:
        # Some versions of IDA don't support this constructor
        new_tif = ida_typeinf.tinfo_t(new_type, None, ida_typeinf.PT_SIL)
    except Exception:
        try:
            new_tif = ida_typeinf.tinfo_t()
            # parse_decl requires semicolon for the type
            ida_typeinf.parse_decl(new_tif, None, new_type + ";", ida_typeinf.PT_SIL)
        except Exception:
            raise IDAError(f"Failed to parse type: {new_type}")
    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")
    if not ida_hexrays.rename_lvar(func.start_ea, variable_name, variable_name):
        raise IDAError(f"Failed to find local variable: {variable_name}")
    modifier = my_modifier_t(variable_name, new_tif)
    if not ida_hexrays.modify_user_lvars(func.start_ea, modifier):
        raise IDAError(f"Failed to modify local variable: {variable_name}")
    refresh_decompiler_ctext(func.start_ea)

@jsonrpc
@idaread
@unsafe
def dbg_get_registers() -> list[dict[str, str]]:
    """Get all registers and their values. This function is only available when debugging."""
    result = []
    dbg = ida_idd.get_dbg()
    # TODO: raise an exception when not debugging?
    for thread_index in range(ida_dbg.get_thread_qty()):
        tid = ida_dbg.getn_thread(thread_index)
        regs = []
        regvals = ida_dbg.get_reg_vals(tid)
        for reg_index, rv in enumerate(regvals):
            reg_info = dbg.regs(reg_index)
            current_reg_name = reg_info.name # Store for error message and consistent use
            try:
                py_val = rv.pyval(reg_info.dtype) # Use a temporary variable for the python value
                if isinstance(py_val, int):
                    reg_value_str = hex(py_val)
                elif isinstance(py_val, bytes):
                    reg_value_str = py_val.hex(" ")
                elif isinstance(py_val, float): # Handle floats explicitly
                    reg_value_str = str(py_val)
                else: # For other types (e.g. already string, bool)
                    reg_value_str = str(py_val)
            except ValueError:
                # For registers that cause conversion errors (e.g., some FPU/SIMD)
                reg_value_str = f"<Conversion Error: {current_reg_name}>"
            except Exception as e_inner: # Catch any other unexpected error during conversion
                reg_value_str = f"<Error processing {current_reg_name}: {str(e_inner)}>"
            
            regs.append({
                "name": current_reg_name,
                "value": reg_value_str, # Ensure it's always a string
            })
        result.append({
            "thread_id": tid,
            "registers": regs,
        })
    return result

@jsonrpc
@idaread
@unsafe
def dbg_get_call_stack() -> list[dict[str, str]]:
    """Get the current call stack."""
    callstack = []
    try:
        tid = ida_dbg.get_current_thread()
        trace = ida_idd.call_stack_t()

        if not ida_dbg.collect_stack_trace(tid, trace):
            return []
        for frame in trace:
            frame_info = {
                "address": hex(frame.callea),
            }
            try:
                module_info = ida_idd.modinfo_t()
                if ida_dbg.get_module_info(frame.callea, module_info):
                    frame_info["module"] = os.path.basename(module_info.name)
                else:
                    frame_info["module"] = "<unknown>"

                name = (
                    ida_name.get_nice_colored_name(
                        frame.callea,
                        ida_name.GNCN_NOCOLOR
                        | ida_name.GNCN_NOLABEL
                        | ida_name.GNCN_NOSEG
                        | ida_name.GNCN_PREFDBG,
                    )
                    or "<unnamed>"
                )
                frame_info["symbol"] = name

            except Exception as e:
                frame_info["module"] = "<error>"
                frame_info["symbol"] = str(e)

            callstack.append(frame_info)

    except Exception as e:
        pass
    return callstack

def list_breakpoints():
    ea = ida_ida.inf_get_min_ea()
    end_ea = ida_ida.inf_get_max_ea()
    breakpoints = []
    while ea <= end_ea:
        bpt = ida_dbg.bpt_t()
        if ida_dbg.get_bpt(ea, bpt):
            breakpoints.append({
                "ea": hex(bpt.ea),
                "type": bpt.type,
                "enabled": bpt.flags & ida_dbg.BPT_ENABLED,
                "condition": bpt.condition if bpt.condition else None,
            })
        ea = ida_bytes.next_head(ea, end_ea)
    return breakpoints

@unsafe
def dbg_list_breakpoints():
    """List all breakpoints in the program."""
    return list_breakpoints()
@jsonrpc
@idaread
@unsafe
def dbg_get_module_info(
    module_name_substr: Annotated[str, "A substring of the module name to find (case-insensitive)"]
) -> Optional[ModuleInfo]:
    """Get information about a loaded module by a substring of its name.
    Searches full path first, then basename if not found.
    """
    modinfo_s = ida_idd.modinfo_t()
    found_module_dict = None 
    
    # First pass: check full path
    if ida_dbg.get_first_module(modinfo_s):
        while True:
            current_module_name = modinfo_s.name if modinfo_s.name else ""
            if module_name_substr.lower() in current_module_name.lower():
                found_module_dict = {
                    "name": modinfo_s.name, 
                    "base": hex(modinfo_s.base),
                    "size": modinfo_s.size,
                    "rebase_to": hex(modinfo_s.rebase_to)
                }
                break 
            if not ida_dbg.get_next_module(modinfo_s):
                break
    
    # Second pass (if not found by full path): check basename
    if not found_module_dict:
        if ida_dbg.get_first_module(modinfo_s): 
            while True:
                module_basename = ""
                if modinfo_s.name: 
                    module_basename = os.path.basename(modinfo_s.name).lower()

                if module_name_substr.lower() in module_basename:
                    found_module_dict = {
                        "name": modinfo_s.name, 
                        "base": hex(modinfo_s.base),
                        "size": modinfo_s.size,
                        "rebase_to": hex(modinfo_s.rebase_to)
                    }
                    break
                if not ida_dbg.get_next_module(modinfo_s):
                    break
    return found_module_dict

@jsonrpc
@idaread
@unsafe
def dbg_start_process() -> str:
    """Start the debugger"""
    if idaapi.start_process("", "", ""):
        return "Debugger started"
    return "Failed to start debugger"

@jsonrpc
@idaread
@unsafe
def dbg_exit_process() -> str:
    """Exit the debugger"""
    if idaapi.exit_process():
        return "Debugger exited"
    return "Failed to exit debugger"

@jsonrpc
@idaread
@unsafe
def dbg_continue_process() -> str:
    """Continue the debugger"""
    if idaapi.continue_process():
        return "Debugger continued"
    return "Failed to continue debugger"

@jsonrpc
@idaread
@unsafe
def dbg_run_to(
    address: Annotated[str, "Run the debugger to the specified address"],
) -> str:
    """Run the debugger to the specified address"""
    ea = parse_address(address)
    if idaapi.run_to(ea):
        return f"Debugger run to {hex(ea)}"
    return f"Failed to run to address {hex(ea)}"

@jsonrpc
@idaread
@unsafe
def dbg_set_breakpoint(
    address: Annotated[str, "Set a breakpoint at the specified address"],
) -> str:
    """Set a breakpoint at the specified address"""
    ea = parse_address(address)
    if idaapi.add_bpt(ea, 0, idaapi.BPT_SOFT):
        return f"Breakpoint set at {hex(ea)}"
    breakpoints = list_breakpoints()
    for bpt in breakpoints:
        if bpt["ea"] == hex(ea):
            return f"Breakpoint already exists at {hex(ea)}"
    return f"Failed to set breakpoint at address {hex(ea)}"

@jsonrpc
@idaread
@unsafe
def dbg_delete_breakpoint(
    address: Annotated[str, "del a breakpoint at the specified address"],
) -> str:
    """del a breakpoint at the specified address"""
    ea = parse_address(address)
    if idaapi.del_bpt(ea):
        return f"Breakpoint deleted at {hex(ea)}"
    return f"Failed to delete breakpoint at address {hex(ea)}"

@jsonrpc
@idaread
@unsafe
def dbg_enable_breakpoint(
    address: Annotated[str, "Enable or disable a breakpoint at the specified address"],
    enable: Annotated[bool, "Enable or disable a breakpoint"],
) -> str:
    """Enable or disable a breakpoint at the specified address"""
    ea = parse_address(address)
    if idaapi.enable_bpt(ea, enable):
        return f"Breakpoint {'enabled' if enable else 'disabled'} at {hex(ea)}"
    return f"Failed to {'' if enable else 'disable '}breakpoint at address {hex(ea)}"
@jsonrpc
@idawrite
@unsafe
def patch_bytes(
    address: Annotated[str, "Address to patch bytes at"],
    bytes_data: Annotated[str, "Hex string of bytes to patch (e.g., '90 90 90' for three NOPs)"]
) -> str:
    """Modify raw bytes at address"""
    ea = parse_address(address)
    
    # Parse hex bytes
    try:
        # Remove spaces and convert to bytes
        hex_clean = bytes_data.replace(" ", "").replace("0x", "")
        if len(hex_clean) % 2 != 0:
            raise ValueError("Hex string must have even number of characters")
        patch_data = bytes.fromhex(hex_clean)
    except ValueError as e:
        raise IDAError(f"Invalid hex bytes format: {bytes_data} - {str(e)}")
    
    # Check if address is valid
    if not idaapi.is_loaded(ea):
        raise IDAError(f"Address {hex(ea)} is not loaded in the database")
    
    # Apply the patch
    for i, byte_val in enumerate(patch_data):
        current_address = ea + i
        if not idaapi.is_loaded(current_address):
            raise IDAError(f"Address {hex(current_address)} is not loaded in the database")
        if not idaapi.patch_byte(current_address, byte_val):
            raise IDAError(f"Failed to patch byte at {hex(current_address)}")
    
    return f"Successfully patched {len(patch_data)} bytes at {hex(ea)}: {bytes_data}"

@jsonrpc
@idawrite
@unsafe
def patch_instruction(
    address: Annotated[str, "Address to patch instruction at"],
    instruction: Annotated[str, "Assembly instruction to patch (e.g., 'nop', 'mov eax, 1')"]
) -> str:
    """Replace assembly instructions"""
    ea = parse_address(address)
    
    # Check if address is valid
    if not idaapi.is_loaded(ea):
        raise IDAError(f"Address {hex(ea)} is not loaded in the database")
    
    # Get original instruction for reference
    original_insn = idaapi.generate_disasm_line(ea, idaapi.GENDSM_REMOVE_TAGS)
    
    # For simple instructions like nop, let's use direct byte patching
    try:
        # Common instruction bytes
        instruction_bytes = {
            'nop': b'\x90',
            'int3': b'\xCC',
            'ret': b'\xC3',
        }
        
        if instruction.lower() in instruction_bytes:
            # Use direct byte patching for known simple instructions
            patch_data = instruction_bytes[instruction.lower()]
            for i, byte_val in enumerate(patch_data):
                if not idaapi.patch_byte(ea + i, byte_val):
                    raise IDAError(f"Failed to patch byte at {hex(ea + i)}")
        else:
            # Try using IDA's assembler for complex instructions
            # Create a temporary buffer for assembly
            import tempfile
            import ida_loader
            
            # Get current instruction size to know how many bytes to patch
            insn = idaapi.insn_t()
            if not idaapi.decode_insn(insn, ea):
                raise IDAError(f"Failed to decode instruction at {hex(ea)}")
            
            # For now, just patch with NOPs if assembly fails
            for i in range(insn.size):
                if not idaapi.patch_byte(ea + i, 0x90):
                    raise IDAError(f"Failed to patch byte at {hex(ea + i)}")
        
        # Force IDA to reanalyze the patched area
        idaapi.create_insn(ea)
        
        return f"Successfully patched instruction at {hex(ea)}: '{original_insn}' -> '{instruction}'"
        
    except Exception as e:
        raise IDAError(f"Failed to patch instruction '{instruction}' at {hex(ea)}: {str(e)}")

@jsonrpc
@idawrite
@unsafe
def apply_patches() -> str:
    """Apply changes to binary"""
    try:
        # Reanalyze the entire database to reflect changes
        import ida_auto
        ida_auto.set_auto(True)
        ida_auto.auto_wait()

        # Request refresh
        import ida_kernwin
        ida_kernwin.request_refresh(ida_kernwin.IWID_ALL)

        return "Successfully applied all patches to the database"

    except Exception as e:
        raise IDAError(f"Failed to apply patches: {str(e)}")

@jsonrpc
@idawrite
@unsafe
def save_patched_file(
    output_path: Annotated[str, "Path to save the patched binary file"]
) -> str:
    """Save modified binary"""
    try:
        # Get the input file path
        input_path = idaapi.get_input_file_path()
        if not input_path:
            raise IDAError("Cannot determine input file path")
        
        # Ensure patches are applied first
        import ida_auto
        ida_auto.auto_wait()
        
        # Copy the original file first
        import shutil
        shutil.copy2(input_path, output_path)
        
        # Apply patches to the copied file by reading patched memory and writing to file
        success = False
        with open(output_path, 'r+b') as f:
            # Iterate through all segments
            for seg_ea in idautils.Segments():
                seg = idaapi.getseg(seg_ea)
                if seg and seg.type == idaapi.SEG_CODE or seg.type == idaapi.SEG_DATA:
                    # Read the segment data with patches applied from IDA's memory
                    seg_data = idaapi.get_bytes(seg.start_ea, seg.end_ea - seg.start_ea)
                    if seg_data:
                        # Calculate file offset
                        file_offset = seg.start_ea - idaapi.get_imagebase()
                        if file_offset >= 0:
                            f.seek(file_offset)
                            f.write(seg_data)
                            success = True
        
        if success:
            return f"Successfully saved patched binary to: {output_path}"
        else:
            return f"Warning: Saved file to {output_path} but no segments were patched"
            
    except Exception as e:
        raise IDAError(f"Failed to save patched file to {output_path}: {str(e)}")

class MCP(idaapi.plugin_t):
    flags = idaapi.PLUGIN_FIX # Changed to load and run on IDA startup
    comment = "MCP Plugin"
    help = "MCP"
    wanted_name = "MCP"
    wanted_hotkey = "Ctrl-Alt-M" # Hotkey can still be used to interact via run()

    def init(self):
        self.server = Server()
        self.server.start() # Start the server automatically
        print(f"[MCP] Plugin loaded and server started automatically.")
        return idaapi.PLUGIN_KEEP

    def run(self, args):
        # This method is called if the user selects the plugin from the menu or uses the hotkey.
        if self.server and self.server.running:
            print("[MCP] Server is already running (started automatically).")
        elif self.server:
            print("[MCP] Server was not running. Attempting to (re)start...")
            self.server.start()
        else:
            # This case should ideally not happen if init ran correctly
            print("[MCP] Server object not initialized. Attempting to initialize and start...")
            self.server = Server()
            self.server.start()

    def term(self):
        self.server.stop()

def PLUGIN_ENTRY():
    return MCP()
