// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/util/memmapped_file_system.proto

#include "tensorflow/core/util/memmapped_file_system.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// This is a temporary google only hack
#ifdef GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
#include "third_party/protobuf/version.h"
#endif
// @@protoc_insertion_point(includes)
namespace tensorflow {
class MemmappedFileSystemDirectoryElementDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<MemmappedFileSystemDirectoryElement>
      _instance;
} _MemmappedFileSystemDirectoryElement_default_instance_;
class MemmappedFileSystemDirectoryDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<MemmappedFileSystemDirectory>
      _instance;
} _MemmappedFileSystemDirectory_default_instance_;
}  // namespace tensorflow
namespace protobuf_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto {
void InitDefaultsMemmappedFileSystemDirectoryElementImpl() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

#ifdef GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
  ::google::protobuf::internal::InitProtobufDefaultsForceUnique();
#else
  ::google::protobuf::internal::InitProtobufDefaults();
#endif  // GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
  {
    void* ptr = &::tensorflow::_MemmappedFileSystemDirectoryElement_default_instance_;
    new (ptr) ::tensorflow::MemmappedFileSystemDirectoryElement();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::tensorflow::MemmappedFileSystemDirectoryElement::InitAsDefaultInstance();
}

void InitDefaultsMemmappedFileSystemDirectoryElement() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &InitDefaultsMemmappedFileSystemDirectoryElementImpl);
}

void InitDefaultsMemmappedFileSystemDirectoryImpl() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

#ifdef GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
  ::google::protobuf::internal::InitProtobufDefaultsForceUnique();
#else
  ::google::protobuf::internal::InitProtobufDefaults();
#endif  // GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
  protobuf_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto::InitDefaultsMemmappedFileSystemDirectoryElement();
  {
    void* ptr = &::tensorflow::_MemmappedFileSystemDirectory_default_instance_;
    new (ptr) ::tensorflow::MemmappedFileSystemDirectory();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::tensorflow::MemmappedFileSystemDirectory::InitAsDefaultInstance();
}

void InitDefaultsMemmappedFileSystemDirectory() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &InitDefaultsMemmappedFileSystemDirectoryImpl);
}

::google::protobuf::Metadata file_level_metadata[2];

const ::google::protobuf::uint32 TableStruct::offsets[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::MemmappedFileSystemDirectoryElement, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::MemmappedFileSystemDirectoryElement, offset_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::MemmappedFileSystemDirectoryElement, name_),
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::MemmappedFileSystemDirectory, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::MemmappedFileSystemDirectory, element_),
};
static const ::google::protobuf::internal::MigrationSchema schemas[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::tensorflow::MemmappedFileSystemDirectoryElement)},
  { 7, -1, sizeof(::tensorflow::MemmappedFileSystemDirectory)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&::tensorflow::_MemmappedFileSystemDirectoryElement_default_instance_),
  reinterpret_cast<const ::google::protobuf::Message*>(&::tensorflow::_MemmappedFileSystemDirectory_default_instance_),
};

void protobuf_AssignDescriptors() {
  AddDescriptors();
  ::google::protobuf::MessageFactory* factory = NULL;
  AssignDescriptors(
      "tensorflow/core/util/memmapped_file_system.proto", schemas, file_default_instances, TableStruct::offsets, factory,
      file_level_metadata, NULL, NULL);
}

void protobuf_AssignDescriptorsOnce() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &protobuf_AssignDescriptors);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_PROTOBUF_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::internal::RegisterAllTypes(file_level_metadata, 2);
}

void AddDescriptorsImpl() {
  InitDefaults();
  static const char descriptor[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
      "\n0tensorflow/core/util/memmapped_file_sy"
      "stem.proto\022\ntensorflow\"C\n#MemmappedFileS"
      "ystemDirectoryElement\022\016\n\006offset\030\001 \001(\004\022\014\n"
      "\004name\030\002 \001(\t\"`\n\034MemmappedFileSystemDirect"
      "ory\022@\n\007element\030\001 \003(\0132/.tensorflow.Memmap"
      "pedFileSystemDirectoryElementB\003\370\001\001b\006prot"
      "o3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 242);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "tensorflow/core/util/memmapped_file_system.proto", &protobuf_RegisterTypes);
}

void AddDescriptors() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &AddDescriptorsImpl);
}
// Force AddDescriptors() to be called at dynamic initialization time.
struct StaticDescriptorInitializer {
  StaticDescriptorInitializer() {
    AddDescriptors();
  }
} static_descriptor_initializer;
}  // namespace protobuf_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto
namespace tensorflow {

// ===================================================================

void MemmappedFileSystemDirectoryElement::InitAsDefaultInstance() {
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int MemmappedFileSystemDirectoryElement::kOffsetFieldNumber;
const int MemmappedFileSystemDirectoryElement::kNameFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

MemmappedFileSystemDirectoryElement::MemmappedFileSystemDirectoryElement()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  if (GOOGLE_PREDICT_TRUE(this != internal_default_instance())) {
    ::protobuf_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto::InitDefaultsMemmappedFileSystemDirectoryElement();
  }
  SharedCtor();
  // @@protoc_insertion_point(constructor:tensorflow.MemmappedFileSystemDirectoryElement)
}
MemmappedFileSystemDirectoryElement::MemmappedFileSystemDirectoryElement(::google::protobuf::Arena* arena)
  : ::google::protobuf::Message(),
  _internal_metadata_(arena) {
  ::protobuf_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto::InitDefaultsMemmappedFileSystemDirectoryElement();
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:tensorflow.MemmappedFileSystemDirectoryElement)
}
MemmappedFileSystemDirectoryElement::MemmappedFileSystemDirectoryElement(const MemmappedFileSystemDirectoryElement& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL),
      _cached_size_(0) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  name_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.name().size() > 0) {
    name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.name(),
      GetArenaNoVirtual());
  }
  offset_ = from.offset_;
  // @@protoc_insertion_point(copy_constructor:tensorflow.MemmappedFileSystemDirectoryElement)
}

void MemmappedFileSystemDirectoryElement::SharedCtor() {
  name_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  offset_ = GOOGLE_ULONGLONG(0);
  _cached_size_ = 0;
}

MemmappedFileSystemDirectoryElement::~MemmappedFileSystemDirectoryElement() {
  // @@protoc_insertion_point(destructor:tensorflow.MemmappedFileSystemDirectoryElement)
  SharedDtor();
}

void MemmappedFileSystemDirectoryElement::SharedDtor() {
  GOOGLE_DCHECK(GetArenaNoVirtual() == NULL);
  name_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}

void MemmappedFileSystemDirectoryElement::ArenaDtor(void* object) {
  MemmappedFileSystemDirectoryElement* _this = reinterpret_cast< MemmappedFileSystemDirectoryElement* >(object);
  (void)_this;
}
void MemmappedFileSystemDirectoryElement::RegisterArenaDtor(::google::protobuf::Arena* arena) {
}
void MemmappedFileSystemDirectoryElement::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* MemmappedFileSystemDirectoryElement::descriptor() {
  ::protobuf_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const MemmappedFileSystemDirectoryElement& MemmappedFileSystemDirectoryElement::default_instance() {
  ::protobuf_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto::InitDefaultsMemmappedFileSystemDirectoryElement();
  return *internal_default_instance();
}


void MemmappedFileSystemDirectoryElement::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.MemmappedFileSystemDirectoryElement)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  name_.ClearToEmpty(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
  offset_ = GOOGLE_ULONGLONG(0);
  _internal_metadata_.Clear();
}

bool MemmappedFileSystemDirectoryElement::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:tensorflow.MemmappedFileSystemDirectoryElement)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // uint64 offset = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(8u /* 8 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint64, ::google::protobuf::internal::WireFormatLite::TYPE_UINT64>(
                 input, &offset_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // string name = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(18u /* 18 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_name()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->name().data(), static_cast<int>(this->name().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "tensorflow.MemmappedFileSystemDirectoryElement.name"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:tensorflow.MemmappedFileSystemDirectoryElement)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:tensorflow.MemmappedFileSystemDirectoryElement)
  return false;
#undef DO_
}

void MemmappedFileSystemDirectoryElement::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:tensorflow.MemmappedFileSystemDirectoryElement)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // uint64 offset = 1;
  if (this->offset() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt64(1, this->offset(), output);
  }

  // string name = 2;
  if (this->name().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->name().data(), static_cast<int>(this->name().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "tensorflow.MemmappedFileSystemDirectoryElement.name");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      2, this->name(), output);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:tensorflow.MemmappedFileSystemDirectoryElement)
}

::google::protobuf::uint8* MemmappedFileSystemDirectoryElement::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.MemmappedFileSystemDirectoryElement)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // uint64 offset = 1;
  if (this->offset() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt64ToArray(1, this->offset(), target);
  }

  // string name = 2;
  if (this->name().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->name().data(), static_cast<int>(this->name().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "tensorflow.MemmappedFileSystemDirectoryElement.name");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        2, this->name(), target);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.MemmappedFileSystemDirectoryElement)
  return target;
}

size_t MemmappedFileSystemDirectoryElement::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.MemmappedFileSystemDirectoryElement)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // string name = 2;
  if (this->name().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->name());
  }

  // uint64 offset = 1;
  if (this->offset() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::UInt64Size(
        this->offset());
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = cached_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void MemmappedFileSystemDirectoryElement::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:tensorflow.MemmappedFileSystemDirectoryElement)
  GOOGLE_DCHECK_NE(&from, this);
  const MemmappedFileSystemDirectoryElement* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const MemmappedFileSystemDirectoryElement>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:tensorflow.MemmappedFileSystemDirectoryElement)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:tensorflow.MemmappedFileSystemDirectoryElement)
    MergeFrom(*source);
  }
}

void MemmappedFileSystemDirectoryElement::MergeFrom(const MemmappedFileSystemDirectoryElement& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.MemmappedFileSystemDirectoryElement)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.name().size() > 0) {
    set_name(from.name());
  }
  if (from.offset() != 0) {
    set_offset(from.offset());
  }
}

void MemmappedFileSystemDirectoryElement::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:tensorflow.MemmappedFileSystemDirectoryElement)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void MemmappedFileSystemDirectoryElement::CopyFrom(const MemmappedFileSystemDirectoryElement& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.MemmappedFileSystemDirectoryElement)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool MemmappedFileSystemDirectoryElement::IsInitialized() const {
  return true;
}

void MemmappedFileSystemDirectoryElement::Swap(MemmappedFileSystemDirectoryElement* other) {
  if (other == this) return;
  if (GetArenaNoVirtual() == other->GetArenaNoVirtual()) {
    InternalSwap(other);
  } else {
    MemmappedFileSystemDirectoryElement* temp = New(GetArenaNoVirtual());
    temp->MergeFrom(*other);
    other->CopyFrom(*this);
    InternalSwap(temp);
    if (GetArenaNoVirtual() == NULL) {
      delete temp;
    }
  }
}
void MemmappedFileSystemDirectoryElement::UnsafeArenaSwap(MemmappedFileSystemDirectoryElement* other) {
  if (other == this) return;
  GOOGLE_DCHECK(GetArenaNoVirtual() == other->GetArenaNoVirtual());
  InternalSwap(other);
}
void MemmappedFileSystemDirectoryElement::InternalSwap(MemmappedFileSystemDirectoryElement* other) {
  using std::swap;
  name_.Swap(&other->name_);
  swap(offset_, other->offset_);
  _internal_metadata_.Swap(&other->_internal_metadata_);
  swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata MemmappedFileSystemDirectoryElement::GetMetadata() const {
  protobuf_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto::file_level_metadata[kIndexInFileMessages];
}


// ===================================================================

void MemmappedFileSystemDirectory::InitAsDefaultInstance() {
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int MemmappedFileSystemDirectory::kElementFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

MemmappedFileSystemDirectory::MemmappedFileSystemDirectory()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  if (GOOGLE_PREDICT_TRUE(this != internal_default_instance())) {
    ::protobuf_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto::InitDefaultsMemmappedFileSystemDirectory();
  }
  SharedCtor();
  // @@protoc_insertion_point(constructor:tensorflow.MemmappedFileSystemDirectory)
}
MemmappedFileSystemDirectory::MemmappedFileSystemDirectory(::google::protobuf::Arena* arena)
  : ::google::protobuf::Message(),
  _internal_metadata_(arena),
  element_(arena) {
  ::protobuf_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto::InitDefaultsMemmappedFileSystemDirectory();
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:tensorflow.MemmappedFileSystemDirectory)
}
MemmappedFileSystemDirectory::MemmappedFileSystemDirectory(const MemmappedFileSystemDirectory& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL),
      element_(from.element_),
      _cached_size_(0) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:tensorflow.MemmappedFileSystemDirectory)
}

void MemmappedFileSystemDirectory::SharedCtor() {
  _cached_size_ = 0;
}

MemmappedFileSystemDirectory::~MemmappedFileSystemDirectory() {
  // @@protoc_insertion_point(destructor:tensorflow.MemmappedFileSystemDirectory)
  SharedDtor();
}

void MemmappedFileSystemDirectory::SharedDtor() {
  GOOGLE_DCHECK(GetArenaNoVirtual() == NULL);
}

void MemmappedFileSystemDirectory::ArenaDtor(void* object) {
  MemmappedFileSystemDirectory* _this = reinterpret_cast< MemmappedFileSystemDirectory* >(object);
  (void)_this;
}
void MemmappedFileSystemDirectory::RegisterArenaDtor(::google::protobuf::Arena* arena) {
}
void MemmappedFileSystemDirectory::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* MemmappedFileSystemDirectory::descriptor() {
  ::protobuf_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const MemmappedFileSystemDirectory& MemmappedFileSystemDirectory::default_instance() {
  ::protobuf_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto::InitDefaultsMemmappedFileSystemDirectory();
  return *internal_default_instance();
}


void MemmappedFileSystemDirectory::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.MemmappedFileSystemDirectory)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  element_.Clear();
  _internal_metadata_.Clear();
}

bool MemmappedFileSystemDirectory::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:tensorflow.MemmappedFileSystemDirectory)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // repeated .tensorflow.MemmappedFileSystemDirectoryElement element = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(10u /* 10 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessage(
                input, add_element()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:tensorflow.MemmappedFileSystemDirectory)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:tensorflow.MemmappedFileSystemDirectory)
  return false;
#undef DO_
}

void MemmappedFileSystemDirectory::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:tensorflow.MemmappedFileSystemDirectory)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .tensorflow.MemmappedFileSystemDirectoryElement element = 1;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->element_size()); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      1,
      this->element(static_cast<int>(i)),
      output);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:tensorflow.MemmappedFileSystemDirectory)
}

::google::protobuf::uint8* MemmappedFileSystemDirectory::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.MemmappedFileSystemDirectory)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .tensorflow.MemmappedFileSystemDirectoryElement element = 1;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->element_size()); i < n; i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      InternalWriteMessageToArray(
        1, this->element(static_cast<int>(i)), deterministic, target);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.MemmappedFileSystemDirectory)
  return target;
}

size_t MemmappedFileSystemDirectory::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.MemmappedFileSystemDirectory)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // repeated .tensorflow.MemmappedFileSystemDirectoryElement element = 1;
  {
    unsigned int count = static_cast<unsigned int>(this->element_size());
    total_size += 1UL * count;
    for (unsigned int i = 0; i < count; i++) {
      total_size +=
        ::google::protobuf::internal::WireFormatLite::MessageSize(
          this->element(static_cast<int>(i)));
    }
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = cached_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void MemmappedFileSystemDirectory::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:tensorflow.MemmappedFileSystemDirectory)
  GOOGLE_DCHECK_NE(&from, this);
  const MemmappedFileSystemDirectory* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const MemmappedFileSystemDirectory>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:tensorflow.MemmappedFileSystemDirectory)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:tensorflow.MemmappedFileSystemDirectory)
    MergeFrom(*source);
  }
}

void MemmappedFileSystemDirectory::MergeFrom(const MemmappedFileSystemDirectory& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.MemmappedFileSystemDirectory)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  element_.MergeFrom(from.element_);
}

void MemmappedFileSystemDirectory::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:tensorflow.MemmappedFileSystemDirectory)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void MemmappedFileSystemDirectory::CopyFrom(const MemmappedFileSystemDirectory& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.MemmappedFileSystemDirectory)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool MemmappedFileSystemDirectory::IsInitialized() const {
  return true;
}

void MemmappedFileSystemDirectory::Swap(MemmappedFileSystemDirectory* other) {
  if (other == this) return;
  if (GetArenaNoVirtual() == other->GetArenaNoVirtual()) {
    InternalSwap(other);
  } else {
    MemmappedFileSystemDirectory* temp = New(GetArenaNoVirtual());
    temp->MergeFrom(*other);
    other->CopyFrom(*this);
    InternalSwap(temp);
    if (GetArenaNoVirtual() == NULL) {
      delete temp;
    }
  }
}
void MemmappedFileSystemDirectory::UnsafeArenaSwap(MemmappedFileSystemDirectory* other) {
  if (other == this) return;
  GOOGLE_DCHECK(GetArenaNoVirtual() == other->GetArenaNoVirtual());
  InternalSwap(other);
}
void MemmappedFileSystemDirectory::InternalSwap(MemmappedFileSystemDirectory* other) {
  using std::swap;
  CastToBase(&element_)->InternalSwap(CastToBase(&other->element_));
  _internal_metadata_.Swap(&other->_internal_metadata_);
  swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata MemmappedFileSystemDirectory::GetMetadata() const {
  protobuf_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto::file_level_metadata[kIndexInFileMessages];
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace tensorflow
namespace google {
namespace protobuf {
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::tensorflow::MemmappedFileSystemDirectoryElement* Arena::CreateMessage< ::tensorflow::MemmappedFileSystemDirectoryElement >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::MemmappedFileSystemDirectoryElement >(arena);
}
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::tensorflow::MemmappedFileSystemDirectory* Arena::CreateMessage< ::tensorflow::MemmappedFileSystemDirectory >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::MemmappedFileSystemDirectory >(arena);
}
}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)
