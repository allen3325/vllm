# Interruptible Inference Architecture Diagrams

## 1. Overall Architecture

```mermaid
graph TB
    subgraph "Engine Core"
        Engine[EngineCore]
        CheckpointMgr[CheckpointManager]
        Engine --> CheckpointMgr
    end

    subgraph "Scheduler Layer"
        Scheduler[Scheduler]
        WaitingQueue[Waiting Queue]
        RunningQueue[Running Queue]
        Scheduler --> WaitingQueue
        Scheduler --> RunningQueue
    end

    subgraph "KV Cache Layer"
        KVManager[KV Cache Manager]
        KVCoordinator[KV Cache Coordinator]
        PrefixCache[Prefix Cache]
        KVManager --> KVCoordinator
        KVCoordinator --> PrefixCache
    end

    subgraph "Worker Layer"
        GPUWorker[GPU Worker]
        ModelRunner[Model Runner]
        CuMemAllocator[CuMem Allocator]
        GPUWorker --> ModelRunner
        GPUWorker --> CuMemAllocator
    end

    Engine --> Scheduler
    Scheduler --> KVManager
    Engine --> GPUWorker

    style CheckpointMgr fill:#8B4545,stroke:#ff6b6b,color:#fff
    style Scheduler fill:#4A5F7F,stroke:#6fa3ef,color:#fff
    style KVManager fill:#4A7C59,stroke:#6fbf73,color:#fff
    style GPUWorker fill:#8B6F47,stroke:#ffb366,color:#fff
```

## 2. State Preservation Flow (Sleep → Wake)

```mermaid
sequenceDiagram
    participant Client
    participant Engine
    participant Scheduler
    participant KVCache
    participant Worker
    participant Checkpoint

    Note over Client,Checkpoint: SLEEP PHASE (preserve_state=True)

    Client->>Engine: sleep(level=2, preserve_state=True)

    Engine->>Scheduler: prepare_for_sleep()
    Note over Scheduler: Preempt all running requests
    Scheduler->>Scheduler: Move running → waiting queue
    Scheduler->>Scheduler: Mark requests as PREEMPTED
    Scheduler-->>Engine: Preemption complete

    Engine->>Scheduler: get_checkpoint_state()
    Scheduler->>Scheduler: Serialize all requests
    Scheduler->>KVCache: get_block_allocations()
    KVCache-->>Scheduler: Block IDs per request
    Scheduler->>KVCache: export_prefix_cache()
    KVCache-->>Scheduler: Prefix cache mappings
    Scheduler-->>Engine: Checkpoint state dict

    Engine->>Checkpoint: save_checkpoint(state)
    Checkpoint->>Checkpoint: Create EngineCheckpoint
    Checkpoint->>Checkpoint: Serialize Request objects
    Checkpoint-->>Engine: Checkpoint saved

    Engine->>Worker: sleep(level=2, preserve_buffers=True)
    Worker->>Worker: Clear model runner cache
    Worker->>Worker: Clear input batch
    Worker->>Worker: Offload to CPU: ("weights", "kv_cache")
    Worker-->>Engine: Sleep complete

    Engine->>Engine: Set is_sleeping = True
    Engine-->>Client: Sleep successful

    Note over Client,Checkpoint: WAKE PHASE

    Client->>Engine: wake_up()

    Engine->>Worker: wake_up()
    Worker->>Worker: Restore from CPU → GPU
    Worker->>Worker: Restore weights memory
    Worker->>Worker: Restore KV cache memory
    Worker-->>Engine: Wake complete

    Engine->>Checkpoint: has_checkpoint?
    Checkpoint-->>Engine: Yes

    Engine->>Checkpoint: load_checkpoint()
    Checkpoint->>Checkpoint: Deserialize EngineCheckpoint
    Checkpoint->>Checkpoint: Recreate Request objects
    Checkpoint-->>Engine: Checkpoint state

    Engine->>Scheduler: restore_checkpoint_state(state)
    Scheduler->>Scheduler: Clear all queues
    Scheduler->>Scheduler: Deserialize requests
    Scheduler->>Scheduler: Restore waiting queue
    Scheduler->>Scheduler: Restore running queue (empty)
    Scheduler->>KVCache: restore_block_allocations(blocks)
    Scheduler->>KVCache: restore_prefix_cache(mappings)
    Scheduler-->>Engine: Restoration complete

    Engine->>Engine: Set is_sleeping = False
    Engine->>Engine: Re-establish block hashers
    Engine-->>Client: Wake complete

    Note over Client,Checkpoint: RESUME INFERENCE

    Engine->>Scheduler: schedule()
    Scheduler->>Scheduler: Pick requests from waiting queue
    Scheduler->>KVCache: allocate_slots(requests)
    Note over KVCache: Blocks already allocated, reuse them
    KVCache-->>Scheduler: Slots ready
    Scheduler->>Scheduler: Move waiting → running
    Scheduler-->>Engine: Batch ready

    Engine->>Worker: execute_model(batch)
    Worker->>Worker: Continue from num_computed_tokens
    Worker->>Worker: Use existing KV cache
    Worker-->>Engine: New tokens generated

    Engine-->>Client: Inference continues seamlessly
```

## 3. Component Interaction Detail

```mermaid
graph LR
    subgraph "CheckpointManager"
        CM[CheckpointManager]
        EC[EngineCheckpoint]
        SC[SchedulerCheckpoint]
        CM --> EC
        EC --> SC
    end

    subgraph "Scheduler State"
        Requests[Requests Map]
        Waiting[Waiting Queue]
        Running[Running Queue]
        Finished[Finished IDs]
    end

    subgraph "KV Cache State"
        BlockAlloc[Block Allocations]
        PCache[Prefix Cache]
        BlockTable[Block Tables]
    end

    subgraph "Request State"
        ReqID[Request ID]
        Tokens[Tokens]
        Params[Sampling Params]
        Status[Status]
        Blocks[Block Hashes]
    end

    SC --> Requests
    SC --> Waiting
    SC --> Running
    SC --> Finished
    SC --> BlockAlloc
    SC --> PCache

    Requests --> ReqID
    Requests --> Tokens
    Requests --> Params
    Requests --> Status
    Requests --> Blocks

    style CM fill:#8B4545,stroke:#ff6b6b,color:#fff
    style SC fill:#4A5F7F,stroke:#6fa3ef,color:#fff
    style BlockAlloc fill:#4A7C59,stroke:#6fbf73,color:#fff
    style PCache fill:#8B6F47,stroke:#ffb366,color:#fff
```

## 4. Request Lifecycle with Interruptible Inference

```mermaid
stateDiagram-v2
    [*] --> Waiting: add_request()

    Waiting --> Running: schedule()
    Running --> Running: continue generation
    Running --> Finished: complete

    Running --> Preempted: sleep(preserve_state=True)
    Preempted --> Checkpointed: save_checkpoint()
    Checkpointed --> Sleeping: offload to CPU

    Sleeping --> Restored: wake_up() + load_checkpoint()
    Restored --> Waiting: restore_checkpoint_state()

    Waiting --> Running: schedule() again
    Running --> Finished: complete generation
    Finished --> [*]

    note right of Preempted
        Running requests are
        moved to waiting queue
        with PREEMPTED status
    end note

    note right of Checkpointed
        All state saved:
        - Request objects
        - KV cache blocks
        - Prefix cache
        - Queue state
    end note

    note right of Restored
        State restored:
        - Deserialize requests
        - Reconnect KV blocks
        - Restore queues
    end note
```

## 5. Memory Layout During Sleep/Wake

```mermaid
graph TB
    subgraph "Before Sleep"
        GPU1[GPU Memory]
        W1[Weights]
        KV1[KV Cache]
        GPU1 --> W1
        GPU1 --> KV1
    end

    subgraph "During Sleep (Level 2, preserve_state=True)"
        CPU[CPU Memory]
        GPU2[GPU Memory]
        WC[Weights Copy]
        KVC[KV Cache Copy]
        CS[Checkpoint State]
        GPU2 -.->|offloaded| CPU
        CPU --> WC
        CPU --> KVC
        CPU --> CS
    end

    subgraph "After Wake"
        GPU3[GPU Memory]
        W3[Weights]
        KV3[KV Cache]
        GPU3 --> W3
        GPU3 --> KV3
        CPU2[CPU Memory]
        CS2[Checkpoint State]
        CPU2 --> CS2
        CPU2 -.->|restored| GPU3
    end

    Before --> During
    During --> After

    style GPU1 fill:#8B4545,stroke:#ff6b6b,color:#fff
    style CPU fill:#4A5F7F,stroke:#6fa3ef,color:#fff
    style GPU3 fill:#4A7C59,stroke:#6fbf73,color:#fff
```

## 6. Decision Flow: preserve_state Parameter

```mermaid
flowchart TD
    Start[Client calls sleep]
    Check{preserve_state?}

    Start --> Check

    Check -->|False<br/>DEFAULT| OldPath[Original Sleep Path]
    OldPath --> OffloadWeights[Offload weights only]
    OffloadWeights --> NoCheckpoint[No checkpoint saved]
    NoCheckpoint --> Sleep1[Enter sleep mode]

    Check -->|True<br/>OPT-IN| NewPath[Interruptible Path]
    NewPath --> HasRequests{Active<br/>requests?}

    HasRequests -->|Yes| Preempt[Preempt all running]
    Preempt --> SaveCheckpoint[Save checkpoint]
    SaveCheckpoint --> OffloadAll[Offload weights + KV cache]
    OffloadAll --> Sleep2[Enter sleep mode]

    HasRequests -->|No| OffloadAll

    Sleep1 --> WakeOld[Wake up - original behavior]
    WakeOld --> RestoreWeights[Restore weights only]
    RestoreWeights --> NoRestore[No state restoration]
    NoRestore --> EndOld[Ready for new requests]

    Sleep2 --> WakeNew[Wake up - with checkpoint]
    WakeNew --> RestoreAll[Restore weights + KV cache]
    RestoreAll --> LoadCheckpoint[Load checkpoint]
    LoadCheckpoint --> RestoreState[Restore scheduler state]
    RestoreState --> ReconnectKV[Reconnect KV blocks]
    ReconnectKV --> EndNew[Resume inference]

    style Check fill:#8B4545,stroke:#ff6b6b,color:#fff
    style NewPath fill:#4A5F7F,stroke:#6fa3ef,color:#fff
    style OldPath fill:#4A7C59,stroke:#6fbf73,color:#fff
    style SaveCheckpoint fill:#8B4545,stroke:#ff6b6b,color:#fff#ff9999
    style LoadCheckpoint fill:#4A5F7F,stroke:#6fa3ef,color:#fff#ff9999
```

## 7. Checkpoint Data Structure

```mermaid
classDiagram
    class EngineCheckpoint {
        +str version
        +float timestamp
        +dict metadata
        +SchedulerCheckpoint scheduler_state
        +to_dict() dict
        +from_dict(dict) EngineCheckpoint
    }

    class SchedulerCheckpoint {
        +dict requests
        +list waiting_queue
        +list running_queue
        +set finished_req_ids
        +dict block_allocations
        +dict prefix_cache
        +to_dict() dict
        +from_dict(dict) SchedulerCheckpoint
    }

    class SerializedRequest {
        +str request_id
        +int client_index
        +list prompt_token_ids
        +list output_token_ids
        +SamplingParams sampling_params
        +int num_computed_tokens
        +int num_cached_tokens
        +list block_hashes
        +RequestStatus status
        +str stop_reason
    }

    class BlockAllocations {
        +str request_id
        +list~int~ block_ids
        +int num_blocks
    }

    class PrefixCacheEntry {
        +tuple hash_key
        +int block_id
        +int ref_count
    }

    EngineCheckpoint *-- SchedulerCheckpoint
    SchedulerCheckpoint *-- SerializedRequest
    SchedulerCheckpoint *-- BlockAllocations
    SchedulerCheckpoint *-- PrefixCacheEntry
```

## 8. Worker Sleep/Wake Flow

```mermaid
flowchart TD
    Start[Worker.sleep called]
    PreserveCheck{preserve_buffers?}

    Start --> PreserveCheck

    PreserveCheck -->|True<br/>Checkpoint mode| ClearCache[Clear model runner cache]
    ClearCache --> ClearBatch[Clear input batch]
    ClearBatch --> OffloadBoth[Offload: weights + kv_cache]

    PreserveCheck -->|False<br/>Original mode| CheckLevel{Level?}
    CheckLevel -->|1| OffloadWeightsOnly[Offload: weights only]
    CheckLevel -->|2| OffloadNone[Offload: nothing]

    OffloadBoth --> CallAllocator1[allocator.sleep]
    OffloadWeightsOnly --> CallAllocator2[allocator.sleep]
    OffloadNone --> CallAllocator3[allocator.sleep]

    CallAllocator1 --> WaitCPU1[Copy to CPU memory]
    CallAllocator2 --> WaitCPU2[Copy to CPU memory]
    CallAllocator3 --> WaitCPU3[Free GPU memory]

    WaitCPU1 --> Complete1[Sleep complete]
    WaitCPU2 --> Complete2[Sleep complete]
    WaitCPU3 --> Complete3[Sleep complete]

    Complete1 --> WakeStart[Worker.wake_up called]
    Complete2 --> WakeStart
    Complete3 --> WakeStart

    WakeStart --> RestoreAllocator[allocator.wake_up]
    RestoreAllocator --> CopyCPUGPU[Copy from CPU → GPU]
    CopyCPUGPU --> RebuildBuffers[Rebuild model buffers]
    RebuildBuffers --> WakeComplete[Wake complete]

    style PreserveCheck fill:#8B4545,stroke:#ff6b6b,color:#fff
    style ClearCache fill:#4A5F7F,stroke:#6fa3ef,color:#fff
    style ClearBatch fill:#4A7C59,stroke:#6fbf73,color:#fff
    style OffloadBoth fill:#4A7C59,stroke:#6fbf73,color:#fff
```

## 9. Error Handling and Edge Cases

```mermaid
flowchart TD
    Start[Operation starts]

    subgraph "Sleep Phase"
        S1[Check if already sleeping]
        S1 -->|Yes| SE1[Log warning, return]
        S1 -->|No| S2[Preempt requests]
        S2 --> S3{Checkpoint save success?}
        S3 -->|No| SE2[Log error, continue sleep]
        S3 -->|Yes| S4[Offload memory]
        S4 --> S5{Offload success?}
        S5 -->|No| SE3[Fatal error, abort]
        S5 -->|Yes| S6[Sleep complete]
    end

    subgraph "Wake Phase"
        W1[Restore memory]
        W1 --> W2{Restore success?}
        W2 -->|No| WE1[Fatal error, abort]
        W2 -->|Yes| W3{Checkpoint exists?}
        W3 -->|No| W4[Log info, skip restore]
        W3 -->|Yes| W5[Load checkpoint]
        W5 --> W6{Deserialize success?}
        W6 -->|No| WE2[Log error, skip restore]
        W6 -->|Yes| W7[Restore state]
        W7 --> W8{Restore success?}
        W8 -->|No| WE3[Log error, partial state]
        W8 -->|Yes| W9[Wake complete]
        W4 --> W9
    end

    subgraph "Resume Phase"
        R1[Schedule requests]
        R1 --> R2{Requests valid?}
        R2 -->|No| RE1[Skip invalid requests]
        R2 -->|Yes| R3[Allocate KV slots]
        R3 --> R4{Allocation success?}
        R4 -->|No| RE2[Preempt request, retry]
        R4 -->|Yes| R5[Execute model]
        RE1 --> R5
        RE2 --> R1
    end

    Start --> S1
    S6 --> W1
    W9 --> R1

    style SE1 fill:#4A7C59,stroke:#6fbf73,color:#fff
    style SE2 fill:#fc932aff
    style SE3 fill:#ff5f5fff
    style WE1 fill:#ff5f5fff
    style WE2 fill:#fc932aff
    style WE3 fill:#fc932aff
    style RE1 fill:#4A7C59,stroke:#6fbf73,color:#fff
    style RE2 fill:#fc932aff

    #ff5f5fff
```

## 10. Comparison: With vs Without preserve_state

```mermaid
graph TB
    subgraph "WITHOUT preserve_state (Original)"
        O1[Request active]
        O2[sleep called]
        O3[Offload weights]
        O4[Request lost]
        O5[wake_up called]
        O6[Restore weights]
        O7[Start fresh]

        O1 --> O2
        O2 --> O3
        O3 --> O4
        O4 --> O5
        O5 --> O6
        O6 --> O7
    end

    subgraph "WITH preserve_state=True (New)"
        N1[Request active]
        N2[sleep called]
        N3[Preempt request]
        N4[Save checkpoint]
        N5[Offload weights + KV]
        N6[wake_up called]
        N7[Restore weights + KV]
        N8[Load checkpoint]
        N9[Resume request]

        N1 --> N2
        N2 --> N3
        N3 --> N4
        N4 --> N5
        N5 --> N6
        N6 --> N7
        N7 --> N8
        N8 --> N9
    end

    style O4 fill:#ff6c6cff
    style N4 fill:#1cc91cff
    style N8 fill:#1cc91cff
    style N9 fill:#1cc91cff
```

## Summary

These diagrams illustrate the complete architecture and flow of the interruptible inference implementation:

1. **Architecture**: Shows the layered component structure
2. **Sequence**: Detailed sleep→wake→resume flow
3. **Component Interaction**: How checkpoint manager coordinates with scheduler and KV cache
4. **Request Lifecycle**: State transitions during sleep/wake
5. **Memory Layout**: GPU↔CPU memory movement
6. **Decision Flow**: How preserve_state parameter affects behavior
7. **Data Structure**: Checkpoint object relationships
8. **Worker Flow**: Low-level worker sleep/wake operations
9. **Error Handling**: Robust error management
10. **Comparison**: Old vs new behavior

The key insight is that **checkpoint-based state preservation** enables true interruptible inference while maintaining full backward compatibility through the opt-in `preserve_state` parameter.
