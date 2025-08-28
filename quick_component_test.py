#!/usr/bin/env python3
"""
Quick Component Test - Verify what's actually working
"""

def test_imports():
    """Test all the main imports"""
    results = {}
    
    # Test mesh_core imports
    try:
        from mesh_core import (
            create_complete_palm_slab,
            PalmSlabInterface, 
            MeshOrchestrator,
            TruthLayer,
            IntentMonitor,
            EmpathyEngine
        )
        results['core_components'] = True
        print("✅ Core components import successfully")
    except Exception as e:
        results['core_components'] = f"❌ {e}"
        print(f"❌ Core components: {e}")
    
    # Test Leon integration
    try:
        from mesh_core.leon_integration import LocalVoiceProcessor
        results['leon_integration'] = True
        print("✅ Leon integration imports successfully")
    except Exception as e:
        results['leon_integration'] = f"❌ {e}"
        print(f"❌ Leon integration: {e}")
    
    # Test Axiom integration
    try:
        from mesh_core.axiom_integration import TruthValidator
        results['axiom_integration'] = True
        print("✅ Axiom integration imports successfully")
    except Exception as e:
        results['axiom_integration'] = f"❌ {e}"
        print(f"❌ Axiom integration: {e}")
    
    # Test memory systems
    try:
        from mesh_core.memory import FactExtractor, KnowledgeGraph
        results['memory_systems'] = True
        print("✅ Memory systems import successfully")
    except Exception as e:
        results['memory_systems'] = f"❌ {e}"
        print(f"❌ Memory systems: {e}")
    
    # Test task management
    try:
        from mesh_core.tasks import TaskParser, WorkflowEngine, TaskScheduler
        results['task_management'] = True
        print("✅ Task management imports successfully")
    except Exception as e:
        results['task_management'] = f"❌ {e}"
        print(f"❌ Task management: {e}")
    
    # Test network components
    try:
        from mesh_core.network import NodeDiscovery, MeshProtocol
        results['network'] = True
        print("✅ Network components import successfully")
    except Exception as e:
        results['network'] = f"❌ {e}"
        print(f"❌ Network: {e}")
    
    return results

def test_palm_slab_creation():
    """Test palm slab creation"""
    try:
        from mesh_core import create_complete_palm_slab
        slab = create_complete_palm_slab()
        print(f"✅ Palm slab created successfully: {type(slab)}")
        print(f"   Available methods: {len([m for m in dir(slab) if not m.startswith('_')])}")
        return True
    except Exception as e:
        print(f"❌ Palm slab creation failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 QUICK COMPONENT TEST")
    print("=" * 50)
    
    print("\n📦 Testing Imports:")
    results = test_imports()
    
    print("\n🌴 Testing Palm Slab:")
    slab_works = test_palm_slab_creation()
    
    print("\n📊 SUMMARY:")
    working = sum(1 for r in results.values() if r is True)
    total = len(results)
    success_rate = (working / total) * 100
    
    print(f"   Components Working: {working}/{total}")
    print(f"   Success Rate: {success_rate:.1f}%")
    print(f"   Palm Slab: {'✅' if slab_works else '❌'}")
    
    if success_rate >= 80:
        print("\n🎉 SYSTEM STATUS: EXCELLENT")
    elif success_rate >= 60:
        print("\n✅ SYSTEM STATUS: GOOD")
    elif success_rate >= 40:
        print("\n⚠️ SYSTEM STATUS: NEEDS WORK")
    else:
        print("\n❌ SYSTEM STATUS: MAJOR ISSUES")