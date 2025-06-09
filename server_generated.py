# NOTE: This file has been automatically generated, do not modify!
# Architecture based on https://github.com/mrexodia/ida-pro-mcp (MIT License)
from typing import Annotated, Optional, TypedDict, Generic, TypeVar
from pydantic import Field

T = TypeVar("T")

class Metadata(TypedDict):
    path: str
    module: str
    base: str
    size: str
    md5: str
    sha256: str
    crc32: str
    filesize: str

class Function(TypedDict):
    address: str
    name: str
    size: str

class ConvertedNumber(TypedDict):
    decimal: str
    hexadecimal: str
    bytes: str
    ascii: Optional[str]
    binary: str

class Page(TypedDict, Generic[T]):
    data: list[T]
    next_offset: Optional[int]

class Global(TypedDict):
    address: str
    name: str

class String(TypedDict):
    address: str
    length: int
    string: str

class ImportedFunction(TypedDict):
    name: str
    address: str

class ImportedModule(TypedDict):
    module_name: str
    functions: list[ImportedFunction]

class FoundBytes(TypedDict):
    address: str

class FoundImmediate(TypedDict):
    address: str
    instruction: str

class Xref(TypedDict):
    address: str
    type: str
    function: Optional[Function]

class ModuleInfo(TypedDict):
    name: str
    base: str
    size: int
    rebase_to: str
    name: str
    base: str
    size: int
    rebase_to: str

@mcp.tool()
def get_metadata() -> Metadata:
    """Get metadata about the current IDB"""
    return make_jsonrpc_request('get_metadata')

@mcp.tool()
def get_function_by_name(name: Annotated[str, Field(description='Name of the function to get')]) -> Function:
    """Get a function by its name"""
    return make_jsonrpc_request('get_function_by_name', name)

@mcp.tool()
def get_function_by_address(address: Annotated[str, Field(description='Address of the function to get')]) -> Function:
    """Get a function by its address"""
    return make_jsonrpc_request('get_function_by_address', address)

@mcp.tool()
def get_current_address() -> str:
    """Get the address currently selected by the user"""
    return make_jsonrpc_request('get_current_address')

@mcp.tool()
def get_current_function() -> Optional[Function]:
    """Get the function currently selected by the user"""
    return make_jsonrpc_request('get_current_function')

@mcp.tool()
def convert_number(text: Annotated[str, Field(description='Textual representation of the number to convert')], size: Annotated[Optional[int], Field(description='Size of the variable in bytes')]) -> ConvertedNumber:
    """Convert a number (decimal, hexadecimal) to different representations"""
    return make_jsonrpc_request('convert_number', text, size)

@mcp.tool()
def list_functions(offset: Annotated[int, Field(description='Offset to start listing from (start at 0)')], count: Annotated[int, Field(description='Number of functions to list (100 is a good default, 0 means remainder)')]) -> Page[Function]:
    """List all functions in the database (paginated)"""
    return make_jsonrpc_request('list_functions', offset, count)

@mcp.tool()
def list_globals_filter(offset: Annotated[int, Field(description='Offset to start listing from (start at 0)')], count: Annotated[int, Field(description='Number of globals to list (100 is a good default, 0 means remainder)')], filter: Annotated[str, Field(description='Filter to apply to the list (required parameter, empty string for no filter). Case-insensitive contains or /regex/ syntax')]) -> Page[Global]:
    """List matching globals in the database (paginated, filtered)"""
    return make_jsonrpc_request('list_globals_filter', offset, count, filter)

@mcp.tool()
def list_globals(offset: Annotated[int, Field(description='Offset to start listing from (start at 0)')], count: Annotated[int, Field(description='Number of globals to list (100 is a good default, 0 means remainder)')]) -> Page[Global]:
    """List all globals in the database (paginated)"""
    return make_jsonrpc_request('list_globals', offset, count)

@mcp.tool()
def list_strings_filter(offset: Annotated[int, Field(description='Offset to start listing from (start at 0)')], count: Annotated[int, Field(description='Number of strings to list (100 is a good default, 0 means remainder)')], filter: Annotated[str, Field(description='Filter to apply to the list (required parameter, empty string for no filter). Case-insensitive contains or /regex/ syntax')]) -> Page[String]:
    """List matching strings in the database (paginated, filtered)"""
    return make_jsonrpc_request('list_strings_filter', offset, count, filter)

@mcp.tool()
def list_strings(offset: Annotated[int, Field(description='Offset to start listing from (start at 0)')], count: Annotated[int, Field(description='Number of strings to list (100 is a good default, 0 means remainder)')]) -> Page[String]:
    """List all strings in the database (paginated)"""
    return make_jsonrpc_request('list_strings', offset, count)

@mcp.tool()
def list_imports() -> list[ImportedModule]:
    """List all imported modules and their functions."""
    return make_jsonrpc_request('list_imports')

@mcp.tool()
def find_bytes(bytes_hex_string: Annotated[str, Field(description="Hex string of bytes to search for (e.g., '43100000')")], search_start_address: Annotated[Optional[str], Field(description='Optional start address for search (hex)')]=None, search_end_address: Annotated[Optional[str], Field(description='Optional end address for search (hex, exclusive)')]=None) -> list[FoundBytes]:
    """Search for a sequence of bytes in the database."""
    return make_jsonrpc_request('find_bytes', bytes_hex_string, search_start_address, search_end_address)

@mcp.tool()
def find_immediate(immediate_value: Annotated[str, Field(description='Immediate value to search for (hex or decimal)')], search_start_address: Annotated[Optional[str], Field(description='Optional start address for search (hex)')]=None, search_end_address: Annotated[Optional[str], Field(description='Optional end address for search (hex, exclusive)')]=None) -> list[FoundImmediate]:
    """Search for an immediate value in code sections."""
    return make_jsonrpc_request('find_immediate', immediate_value, search_start_address, search_end_address)

@mcp.tool()
def decompile_function(address: Annotated[str, Field(description='Address of the function to decompile')]) -> str:
    """Decompile a function at the given address"""
    return make_jsonrpc_request('decompile_function', address)

@mcp.tool()
def disassemble_function(start_address: Annotated[str, Field(description='Address of the function to disassemble')]) -> str:
    """Get assembly code (address: instruction; comment) for a function"""
    return make_jsonrpc_request('disassemble_function', start_address)

@mcp.tool()
def get_xrefs_to(address: Annotated[str, Field(description='Address to get cross references to')]) -> list[Xref]:
    """Get all cross references to the given address"""
    return make_jsonrpc_request('get_xrefs_to', address)

@mcp.tool()
def get_xrefs_to_field(struct_name: Annotated[str, Field(description='Name of the struct (type) containing the field')], field_name: Annotated[str, Field(description='Name of the field (member) to get xrefs to')]) -> list[Xref]:
    """Get all cross references to a named struct field (member)"""
    return make_jsonrpc_request('get_xrefs_to_field', struct_name, field_name)

@mcp.tool()
def get_entry_points() -> list[Function]:
    """Get all entry points in the database"""
    return make_jsonrpc_request('get_entry_points')

@mcp.tool()
def set_comment(address: Annotated[str, Field(description='Address in the function to set the comment for')], comment: Annotated[str, Field(description='Comment text')]):
    """Set a comment for a given address in the function disassembly and pseudocode"""
    return make_jsonrpc_request('set_comment', address, comment)

@mcp.tool()
def rename_local_variable(function_address: Annotated[str, Field(description='Address of the function containing the variable')], old_name: Annotated[str, Field(description='Current name of the variable')], new_name: Annotated[str, Field(description='New name for the variable (empty for a default name)')]):
    """Rename a local variable in a function"""
    return make_jsonrpc_request('rename_local_variable', function_address, old_name, new_name)

@mcp.tool()
def rename_global_variable(old_name: Annotated[str, Field(description='Current name of the global variable')], new_name: Annotated[str, Field(description='New name for the global variable (empty for a default name)')]):
    """Rename a global variable"""
    return make_jsonrpc_request('rename_global_variable', old_name, new_name)

@mcp.tool()
def set_global_variable_type(variable_name: Annotated[str, Field(description='Name of the global variable')], new_type: Annotated[str, Field(description='New type for the variable')]):
    """Set a global variable's type"""
    return make_jsonrpc_request('set_global_variable_type', variable_name, new_type)

@mcp.tool()
def rename_function(function_address: Annotated[str, Field(description='Address of the function to rename')], new_name: Annotated[str, Field(description='New name for the function (empty for a default name)')]):
    """Rename a function"""
    return make_jsonrpc_request('rename_function', function_address, new_name)

@mcp.tool()
def set_function_prototype(function_address: Annotated[str, Field(description='Address of the function')], prototype: Annotated[str, Field(description='New function prototype')]) -> str:
    """Set a function's prototype"""
    return make_jsonrpc_request('set_function_prototype', function_address, prototype)

@mcp.tool()
def declare_c_type(c_declaration: Annotated[str, Field(description='C declaration of the type. Examples include: typedef int foo_t; struct bar { int a; bool b; };')]):
    """Create or update a local type from a C declaration"""
    return make_jsonrpc_request('declare_c_type', c_declaration)

@mcp.tool()
def set_local_variable_type(function_address: Annotated[str, Field(description='Address of the function containing the variable')], variable_name: Annotated[str, Field(description='Name of the variable')], new_type: Annotated[str, Field(description='New type for the variable')]):
    """Set a local variable's type"""
    return make_jsonrpc_request('set_local_variable_type', function_address, variable_name, new_type)

@mcp.tool()
def dbg_get_registers() -> list[dict[str, str]]:
    """Get all registers and their values. This function is only available when debugging."""
    return make_jsonrpc_request('dbg_get_registers')

@mcp.tool()
def dbg_get_call_stack() -> list[dict[str, str]]:
    """Get the current call stack."""
    return make_jsonrpc_request('dbg_get_call_stack')

@mcp.tool()
def dbg_get_module_info(module_name_substr: Annotated[str, Field(description='A substring of the module name to find (case-insensitive)')]) -> Optional[ModuleInfo]:
    """Get information about a loaded module by a substring of its name.
    Searches full path first, then basename if not found.
    """
    return make_jsonrpc_request('dbg_get_module_info', module_name_substr)

@mcp.tool()
def dbg_start_process() -> str:
    """Start the debugger"""
    return make_jsonrpc_request('dbg_start_process')

@mcp.tool()
def dbg_exit_process() -> str:
    """Exit the debugger"""
    return make_jsonrpc_request('dbg_exit_process')

@mcp.tool()
def dbg_continue_process() -> str:
    """Continue the debugger"""
    return make_jsonrpc_request('dbg_continue_process')

@mcp.tool()
def dbg_run_to(address: Annotated[str, Field(description='Run the debugger to the specified address')]) -> str:
    """Run the debugger to the specified address"""
    return make_jsonrpc_request('dbg_run_to', address)

@mcp.tool()
def dbg_set_breakpoint(address: Annotated[str, Field(description='Set a breakpoint at the specified address')]) -> str:
    """Set a breakpoint at the specified address"""
    return make_jsonrpc_request('dbg_set_breakpoint', address)

@mcp.tool()
def dbg_delete_breakpoint(address: Annotated[str, Field(description='del a breakpoint at the specified address')]) -> str:
    """del a breakpoint at the specified address"""
    return make_jsonrpc_request('dbg_delete_breakpoint', address)

@mcp.tool()
def dbg_enable_breakpoint(address: Annotated[str, Field(description='Enable or disable a breakpoint at the specified address')], enable: Annotated[bool, Field(description='Enable or disable a breakpoint')]) -> str:
    """Enable or disable a breakpoint at the specified address"""
    return make_jsonrpc_request('dbg_enable_breakpoint', address, enable)

@mcp.tool()
def patch_bytes(address: Annotated[str, Field(description='Address to patch bytes at')], bytes_data: Annotated[str, Field(description="Hex string of bytes to patch (e.g., '90 90 90' for three NOPs)")]) -> str:
    """Modify raw bytes at address"""
    return make_jsonrpc_request('patch_bytes', address, bytes_data)

@mcp.tool()
def patch_instruction(address: Annotated[str, Field(description='Address to patch instruction at')], instruction: Annotated[str, Field(description="Assembly instruction to patch (e.g., 'nop', 'mov eax, 1')")]) -> str:
    """Replace assembly instructions"""
    return make_jsonrpc_request('patch_instruction', address, instruction)

@mcp.tool()
def apply_patches() -> str:
    """Apply changes to binary"""
    return make_jsonrpc_request('apply_patches')

@mcp.tool()
def save_patched_file(output_path: Annotated[str, Field(description='Path to save the patched binary file')]) -> str:
    """Save modified binary"""
    return make_jsonrpc_request('save_patched_file', output_path)

