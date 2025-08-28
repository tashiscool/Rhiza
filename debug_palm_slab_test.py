#!/usr/bin/env python3
"""Debug test for palm slab creation issues"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def debug_palm_slab():
    """Debug palm slab creation and initialization"""
    
    try:
        print("🔧 Debugging Palm Slab Creation")
        print("=" * 40)
        
        # Try direct import
        print("1. Testing direct PalmSlabInterface import...")
        from src.mesh_core.palm_slab_interface import PalmSlabInterface
        palm_slab = PalmSlabInterface(node_id="debug_test", privacy_level="selective")
        print(f"   ✅ Direct creation successful: {palm_slab}")
        print(f"   📊 Node ID: {palm_slab.node_id}")
        print(f"   🔒 Privacy Level: {palm_slab.privacy_level}")
        
        # Check if initialize method exists
        print(f"   🔍 Has initialize method: {hasattr(palm_slab, 'initialize')}")
        print(f"   🔍 Initialize method type: {type(getattr(palm_slab, 'initialize', None))}")
        
        # Try initialization
        print("\n2. Testing initialization...")
        try:
            await palm_slab.initialize()
            print("   ✅ Initialization successful")
            
            # Check status
            if hasattr(palm_slab, 'get_palm_slab_status'):
                status = await palm_slab.get_palm_slab_status()
                print(f"   📊 Status: {status}")
            else:
                print("   ⚠️  No get_palm_slab_status method")
                
        except Exception as init_error:
            print(f"   ❌ Initialization failed: {init_error}")
            import traceback
            traceback.print_exc()
        
        # Try factory function
        print("\n3. Testing factory function...")
        from src.mesh_core.palm_slab_interface import create_palm_slab_node
        palm_slab2 = create_palm_slab_node(node_id="debug_test2", privacy_level="selective")
        print(f"   ✅ Factory creation successful: {palm_slab2}")
        
        # Try mesh_core import
        print("\n4. Testing mesh_core import...")
        from mesh_core import create_complete_palm_slab
        palm_slab3 = create_complete_palm_slab(node_id="debug_test3", privacy_level="selective")
        print(f"   ✅ Mesh core creation successful: {palm_slab3}")
        
        # Check attributes
        print(f"\n5. Attribute comparison:")
        print(f"   Direct: {dir(palm_slab)[:5]}...")
        print(f"   Factory: {dir(palm_slab2)[:5]}...")
        print(f"   Mesh Core: {dir(palm_slab3)[:5]}...")
        
    except Exception as e:
        print(f"❌ Debug test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_palm_slab())